import numpy as np
from pathlib import Path

N_AU_FEATURES = 40

EMOTION_TO_AU = {
    "angry":    [0.3,0.0,0.8,0.2,0.0,0.7,0.0,0.0,0.0,0.0,0.3,0.3,0.5,0.3,0.8,0.0,0.0,0.7,0.0,0.1],
    "disgust":  [0.0,0.0,0.5,0.0,0.0,0.8,0.5,0.0,0.0,0.0,0.3,0.0,0.3,0.2,0.5,0.0,0.0,0.3,0.0,0.1],
    "fear":     [0.7,0.7,0.0,0.7,0.0,0.0,0.0,0.0,0.0,0.3,0.2,0.7,0.5,0.3,0.0,0.0,0.3,0.2,0.0,0.3],
    "happy":    [0.3,0.3,0.0,0.3,0.8,0.0,0.0,0.9,0.3,0.0,0.0,0.0,0.5,0.2,0.0,0.9,0.0,0.0,0.0,0.2],
    "sad":      [0.6,0.0,0.5,0.0,0.0,0.0,0.0,0.0,0.3,0.6,0.5,0.0,0.2,0.3,0.5,0.0,0.6,0.2,0.0,0.1],
    "surprise": [0.7,0.8,0.0,0.8,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.8,0.7,0.0,0.0,0.0,0.0,0.0,0.4],
    "neutral":  [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.2],
}


def extract_deepface_features(frame_paths):

    if not frame_paths:
        return np.zeros(N_AU_FEATURES, dtype=np.float32)
    try:
        from deepface import DeepFace
        all_rows = []
        for fpath in frame_paths:
            if not Path(str(fpath)).exists():
                continue
            try:
                result   = DeepFace.analyze(str(fpath), actions=["emotion"],
                                            enforce_detection=False, silent=True)
                emo_dict = result[0]["emotion"] if isinstance(result, list) else result["emotion"]
                au_vec   = np.zeros(20, dtype=np.float32)
                for emo, prob in emo_dict.items():
                    emo_l = emo.lower()
                    if emo_l in EMOTION_TO_AU:
                        au_vec += np.array(EMOTION_TO_AU[emo_l], dtype=np.float32) * (prob / 100.0)
                all_rows.append(au_vec)
            except Exception:
                continue
        if not all_rows:
            return np.zeros(N_AU_FEATURES, dtype=np.float32)
        arr = np.array(all_rows, dtype=np.float32)
        return np.concatenate([arr.mean(axis=0), arr.std(axis=0)]).astype(np.float32)
    except ImportError:
        print("[WARN] DeepFace not installed.")
        return np.zeros(N_AU_FEATURES, dtype=np.float32)
    except Exception as e:
        print(f"[WARN] DeepFace error: {e}")
        return np.zeros(N_AU_FEATURES, dtype=np.float32)
