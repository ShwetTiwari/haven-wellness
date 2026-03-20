import sys
import uuid
import base64
import json
import joblib
import tempfile
import numpy as np
from pathlib import Path
from flask import Flask, request, jsonify, render_template, session, Response
from flask_cors import CORS
from werkzeug.utils import secure_filename

BASE_DIR   = Path(__file__).parent.parent
MODELS_DIR = BASE_DIR / "models" / "saved"
UPLOAD_DIR = BASE_DIR / "data" / "uploads"
FRAMES_DIR = UPLOAD_DIR / "frames"
AUDIO_DIR  = UPLOAD_DIR / "audio"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
FRAMES_DIR.mkdir(parents=True, exist_ok=True)
AUDIO_DIR.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(BASE_DIR))

from config import SECRET_KEY, PORT, DEBUG

app = Flask(
    __name__,
    template_folder=str(BASE_DIR / "frontend" / "templates"),
    static_folder=str(BASE_DIR / "frontend" / "static")
)
app.secret_key = SECRET_KEY
CORS(app, supports_credentials=True)
app.config["MAX_CONTENT_LENGTH"] = 500 * 1024 * 1024  # 500MB for full audio

print("[Haven] Loading models...")
try:
    MODEL     = joblib.load(MODELS_DIR / "ensemble_model.pkl")
    EXTRACTOR = joblib.load(MODELS_DIR / "feature_extractor.pkl")
    print("[Haven] Models loaded!")
    MODELS_LOADED = True
except FileNotFoundError:
    print("[Haven] Models not found - run train_model.py first.")
    MODEL, EXTRACTOR = None, None
    MODELS_LOADED = False


def wellness_label(prob):
    if prob < 0.35: return "Thriving"
    if prob < 0.55: return "Balancing"
    if prob < 0.75: return "Recharging"
    return "Seeking support"


def save_b64_frame(b64, idx, sid):
    try:
        data  = b64.split(",")[1] if "," in b64 else b64
        path  = FRAMES_DIR / f"{sid}_{idx:04d}.jpg"
        with open(str(path), "wb") as f:
            f.write(base64.b64decode(data))
        return str(path)
    except Exception:
        return None


def cleanup(*paths):
    for p in paths:
        if p:
            try: Path(p).unlink(missing_ok=True)
            except Exception: pass


def merge_audio_chunks(chunks_b64: list, session_id: str) -> str:
    """
    Merges multiple base64 WebM/WAV audio chunks into a single WAV file.
    Returns path to merged WAV.
    """
    if not chunks_b64:
        return None
    try:
        import soundfile as sf
        import io

        all_samples = []
        target_sr   = 16000

        for i, chunk_b64 in enumerate(chunks_b64):
            try:
                data = chunk_b64.split(",")[1] if "," in chunk_b64 else chunk_b64
                raw  = base64.b64decode(data)
                # Try reading with soundfile
                with io.BytesIO(raw) as buf:
                    samples, sr = sf.read(buf, dtype='float32')
                    if samples.ndim > 1:
                        samples = samples.mean(axis=1)  # stereo -> mono
                    # Resample to 16kHz if needed
                    if sr != target_sr:
                        import librosa
                        samples = librosa.resample(samples, orig_sr=sr, target_sr=target_sr)
                    all_samples.append(samples)
            except Exception as e:
                # Try with pydub as fallback
                try:
                    import librosa
                    import io as _io
                    y, sr = librosa.load(_io.BytesIO(raw), sr=target_sr, mono=True)
                    all_samples.append(y)
                except Exception:
                    print(f"  [WARN] Could not decode audio chunk {i}: {e}")
                    continue

        if not all_samples:
            return None

        merged   = np.concatenate(all_samples)
        out_path = str(AUDIO_DIR / f"{session_id}_merged.wav")
        sf.write(out_path, merged, target_sr)
        print(f"[Audio] Merged {len(all_samples)} chunks → {len(merged)/target_sr:.1f}s WAV")
        return out_path

    except Exception as e:
        print(f"[Audio] Merge error: {e}")
        return None


# ── TTS ───────────────────────────────────────────────────

@app.route("/tts", methods=["POST"])
def tts():
    data = request.get_json() or {}
    text = data.get("text", "").strip()
    if not text:
        return "", 204
    try:
        from tts_engine import synthesize_speech
        audio = synthesize_speech(text)
        if not audio:
            return "", 204
        return Response(audio, mimetype="audio/mpeg",
                        headers={"Content-Disposition": "inline"})
    except Exception as e:
        print(f"[TTS] Error: {e}")
        return "", 204


# ── Main routes ───────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/health")
def health():
    return jsonify({"status": "ok", "models_loaded": MODELS_LOADED})


@app.route("/remi/start", methods=["POST"])
def remi_start():
    session["conversation"]   = []
    session["user_texts"]     = []
    session["session_id"]     = uuid.uuid4().hex
    session["audio_chunks"]   = []   # VAD audio chunk list
    session["frame_count"]    = 0
    opener = "Hey! Really glad you're here."
    card   = {
        "type":   "emoji_mood",
        "emojis": ["😄","🙂","😐","😔","😞"],
        "labels": ["Great","Good","Okay","Low","Rough"],
    }
    session["conversation"] = [{"role": "assistant", "content": opener}]
    return jsonify({
        "message": opener, "tts_text": opener,
        "phase": 1, "analysis_ready": False, "card": card,
    })


@app.route("/remi/chat", methods=["POST"])
def remi_chat():
    data = request.get_json() or {}
    msg  = data.get("message", "").strip()
    if not msg:
        return jsonify({"error": "Empty message"}), 400

    conversation = session.get("conversation", [])
    user_texts   = session.get("user_texts", [])
    user_texts.append(msg)
    conversation.append({"role": "user", "content": msg})

    from remi_chat import get_remi_response
    result = get_remi_response(conversation[:-1], msg)
    conversation.append({"role": "assistant", "content": result["message"]})

    session["conversation"] = conversation
    session["user_texts"]   = user_texts
    return jsonify(result)


@app.route("/remi/signal", methods=["POST"])
def remi_signal():
    """Passive VADER sentiment signal for ambient orb."""
    data = request.get_json() or {}
    text = data.get("text", "")
    if not text.strip():
        return jsonify({"signal": 0.5})
    try:
        from nltk.sentiment.vader import SentimentIntensityAnalyzer
        sia    = SentimentIntensityAnalyzer()
        score  = sia.polarity_scores(text)
        signal = (score["compound"] + 1) / 2
        return jsonify({"signal": round(signal, 3)})
    except Exception:
        return jsonify({"signal": 0.5})


@app.route("/remi/audio_chunk", methods=["POST"])
def audio_chunk():
    """
    Receives a VAD-detected audio chunk (base64 encoded WebM/WAV).
    Stores in session for later merging.
    """
    data  = request.get_json() or {}
    chunk = data.get("chunk", "")
    if not chunk:
        return jsonify({"ok": False})

    chunks = session.get("audio_chunks", [])
    chunks.append(chunk)
    session["audio_chunks"] = chunks
    return jsonify({"ok": True, "total_chunks": len(chunks)})


@app.route("/remi/frame", methods=["POST"])
def receive_frame():
    """
    Receives a webcam frame (base64 JPEG).
    Stores path in session.
    """
    data  = request.get_json() or {}
    frame = data.get("frame", "")
    if not frame:
        return jsonify({"ok": False})

    sid    = session.get("session_id", uuid.uuid4().hex)
    count  = session.get("frame_count", 0)
    path   = save_b64_frame(frame, count, sid)
    if path:
        session["frame_count"] = count + 1
    return jsonify({"ok": bool(path), "frame_count": session["frame_count"]})


@app.route("/remi/analyze", methods=["POST"])
def remi_analyze():
    """
    Final analysis after conversation ends.
    Uses all accumulated audio chunks + frames + text.
    """
    sid          = session.get("session_id", uuid.uuid4().hex)
    user_texts   = session.get("user_texts", [])
    audio_chunks = session.get("audio_chunks", [])
    frame_count  = session.get("frame_count", 0)
    full_text    = " ".join(user_texts)
    merged_wav   = None
    frame_paths  = []

    try:
        # Collect stored frames
        for i in range(frame_count):
            p = FRAMES_DIR / f"{sid}_{i:04d}.jpg"
            if p.exists():
                frame_paths.append(str(p))

        # Also accept frames sent in the final request
        extra_frames = json.loads(request.form.get("frames", "[]"))
        for i, b64 in enumerate(extra_frames):
            fp = save_b64_frame(b64, frame_count + i, sid)
            if fp: frame_paths.append(fp)

        # Merge VAD audio chunks into single WAV
        if audio_chunks:
            print(f"[Analyze] Merging {len(audio_chunks)} audio chunks...")
            merged_wav = merge_audio_chunks(audio_chunks, sid)

        if not MODELS_LOADED:
            import random
            prob = random.uniform(0.2, 0.7)
            from remi_chat import get_wellness_tips
            return jsonify({
                "wellness_label":  wellness_label(prob),
                "tips":            get_wellness_tips(prob),
                "vibe_score":      round((1 - prob) * 100),
                "modality_scores": {"text": 0.5, "audio": 0.5, "video": 0.5},
                "inputs_used":     {"text": bool(full_text), "audio": bool(merged_wav), "video": bool(frame_paths)},
                "demo": True,
            })

        print(f"[Analyze] Text: {len(full_text)} chars | "
              f"Audio: {merged_wav or 'none'} | Frames: {len(frame_paths)}")

        # Feature extraction using live path (same space as training)
        X = EXTRACTOR.transform_live(
            text=full_text,
            audio_path=merged_wav,
            frame_paths=frame_paths if frame_paths else None
        )

        prob  = float(MODEL.predict_proba(X)[0][1])
        n_t   = EXTRACTOR.text_ext.transform([full_text]).shape[1]
        n_a   = EXTRACTOR.audio_ext.n_features
        n_v   = EXTRACTOR.video_ext.n_features

        def mp(mt=False, ma=False, mv=False):
            Xm = X.copy()
            if mt: Xm[:, :n_t] = 0
            if ma: Xm[:, n_t:n_t+n_a] = 0
            if mv: Xm[:, n_t+n_a:] = 0
            return round(float(MODEL.predict_proba(Xm)[0][1]), 3)

        from remi_chat import get_wellness_tips
        return jsonify({
            "wellness_label":  wellness_label(prob),
            "tips":            get_wellness_tips(prob),
            "vibe_score":      round((1 - prob) * 100),
            "modality_scores": {
                "text":  mp(ma=True, mv=True),
                "audio": mp(mt=True, mv=True),
                "video": mp(mt=True, ma=True),
            },
            "inputs_used": {
                "text":  bool(full_text),
                "audio": bool(merged_wav),
                "video": bool(frame_paths),
            },
        })

    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({"error": str(e)}), 500
    finally:
        if merged_wav: cleanup(merged_wav)
        for fp in frame_paths: cleanup(fp)


# ── Wellness endpoints ────────────────────────────────────

@app.route("/wellness/meme")
def get_meme():
    import random
    memes = [
        {"top":"me at 11pm","bottom":"deciding to fix my sleep schedule starting tomorrow"},
        {"top":"my brain at 3am","bottom":"remember that embarrassing thing from 2014"},
        {"top":"therapist: and how does that make you feel","bottom":"me: ...hungry"},
        {"top":"self care is","bottom":"saying no to things that drain you"},
        {"top":"rest is not laziness","bottom":"your body is not a machine"},
        {"top":"you survived 100% of your bad days","bottom":"that's a pretty good track record"},
        {"top":"it's okay to not be okay","bottom":"it's also okay to eat a snack about it"},
        {"top":"healing is not linear","bottom":"neither is my sleep schedule"},
        {"top":"gentle reminder","bottom":"you don't have to earn your rest"},
        {"top":"plot twist","bottom":"the main character needed a nap the whole time"},
        {"top":"not everything needs to be productive","bottom":"sometimes existing is enough"},
        {"top":"you don't have to be okay all the time","bottom":"that's literally not how humans work"},
    ]
    return jsonify(random.choice(memes))


@app.route("/wellness/quote")
def get_quote():
    import random
    quotes = [
        {"text":"You don't have to be positive all the time. It's perfectly okay to feel sad, angry, annoyed, frustrated, scared or anxious.","author":"Lori Deschene"},
        {"text":"Almost everything will work again if you unplug it for a few minutes, including you.","author":"Anne Lamott"},
        {"text":"You are allowed to be both a masterpiece and a work in progress simultaneously.","author":"Sophia Bush"},
        {"text":"There is hope, even when your brain tells you there isn't.","author":"John Green"},
        {"text":"You don't have to control your thoughts. You just have to stop letting them control you.","author":"Dan Millman"},
        {"text":"Sometimes the most important thing in a whole day is the rest we take between two deep breaths.","author":"Etty Hillesum"},
        {"text":"You are enough. You have always been enough. You will always be enough.","author":"Unknown"},
        {"text":"Be gentle with yourself. You are a child of the universe, no less than the trees and the stars.","author":"Max Ehrmann"},
    ]
    return jsonify(random.choice(quotes))


@app.route("/wellness/affirmation")
def get_affirmation():
    import random
    affirms = [
        "I am allowed to take up space.",
        "My feelings are valid, even when I can't explain them.",
        "I am doing the best I can with what I have right now.",
        "Rest is productive. I am allowed to rest.",
        "I don't have to have it all figured out today.",
        "I am worthy of kindness — including from myself.",
        "It's okay to ask for help. That's strength, not weakness.",
        "I have survived difficult things before. I can do it again.",
        "My worth is not measured by my productivity.",
        "Small steps forward are still progress.",
        "Today I choose to be gentle with myself.",
        "I am allowed to change my mind and grow.",
    ]
    return jsonify({"text": random.choice(affirms)})


if __name__ == "__main__":
    print("\n" + "="*50)
    print("  Haven - Full Multimodal Companion")
    print(f"  http://localhost:{PORT}")
    print("="*50 + "\n")
    app.run(debug=DEBUG, host="0.0.0.0", port=PORT)