import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from textblob import TextBlob
from sklearn.feature_extraction.text import TfidfVectorizer
import warnings
warnings.filterwarnings("ignore")

for pkg in ["vader_lexicon", "punkt", "stopwords", "punkt_tab"]:
    try: nltk.download(pkg, quiet=True)
    except Exception: pass

# Audio config - same for training AND live inference
N_MFCC      = 40
N_CHROMA    = 12
N_MEL       = 20
AUDIO_SR    = 16000
AUDIO_DUR   = 180   # max seconds to load from training files

# AU columns from CLNF (used only for training data, not live)
AU_R = ["AU01_r","AU02_r","AU04_r","AU05_r","AU06_r","AU09_r",
        "AU10_r","AU12_r","AU14_r","AU15_r","AU17_r","AU20_r","AU25_r","AU26_r"]
AU_C = ["AU04_c","AU12_c","AU15_c","AU23_c","AU28_c","AU45_c"]
AU_COLS = AU_R + AU_C  # 20 AUs


def extract_audio_features_from_wav(wav_path, sr=AUDIO_SR, duration=AUDIO_DUR) -> np.ndarray:
    """
    Extract unified audio features from a WAV file using librosa.
    Used for BOTH training (from DAIC-WOZ WAV files) and live inference.
    Returns: 144-dim feature vector (mean+std for MFCC, chroma, mel)
    """
    n_out = (N_MFCC + N_CHROMA + N_MEL) * 2  # 144
    if not wav_path or not Path(str(wav_path)).exists():
        return np.zeros(n_out, dtype=np.float32)
    try:
        import librosa
        y, sr_loaded = librosa.load(str(wav_path), sr=sr, duration=duration, mono=True)
        if len(y) < sr * 0.5:  # less than 0.5s
            return np.zeros(n_out, dtype=np.float32)

        # MFCCs
        mfcc    = librosa.feature.mfcc(y=y, sr=sr_loaded, n_mfcc=N_MFCC)
        # Chroma
        chroma  = librosa.feature.chroma_stft(y=y, sr=sr_loaded, n_chroma=N_CHROMA)
        # Mel spectrogram
        mel     = librosa.feature.melspectrogram(y=y, sr=sr_loaded, n_mels=N_MEL)
        mel_db  = librosa.power_to_db(mel, ref=np.max)

        feats = np.concatenate([
            mfcc.mean(axis=1),   mfcc.std(axis=1),
            chroma.mean(axis=1), chroma.std(axis=1),
            mel_db.mean(axis=1), mel_db.std(axis=1),
        ]).astype(np.float32)

        return feats

    except Exception as e:
        print(f"  [WARN] Audio error {wav_path}: {e}")
        n_out = (N_MFCC + N_CHROMA + N_MEL) * 2
        return np.zeros(n_out, dtype=np.float32)


# ── TEXT ─────────────────────────────────────────────────────────────────────

class TextFeatureExtractor:
    N_TFIDF = 50

    def __init__(self):
        self.vader   = SentimentIntensityAnalyzer()
        self.tfidf   = TfidfVectorizer(max_features=self.N_TFIDF,
                                        stop_words="english", ngram_range=(1,2), min_df=1)
        self._fitted = False

    def _scalar(self, text):
        text = text.strip() if text else ""
        vs   = self.vader.polarity_scores(text) if text else {"compound":0,"pos":0,"neg":0,"neu":1}
        blob = TextBlob(text) if text else None
        pol  = blob.sentiment.polarity     if blob else 0
        sub  = blob.sentiment.subjectivity if blob else 0
        words = text.split()
        return np.array([
            vs["compound"], vs["pos"], vs["neg"], vs["neu"], pol, sub,
            len(words), len(set(words))/max(len(words),1),
            float(np.mean([len(w) for w in words])) if words else 0,
            max(len([s for s in text.split(".") if s.strip()]), 1),
        ], dtype=np.float32)

    def fit(self, texts):
        self.tfidf.fit([t if t.strip() else "empty" for t in texts])
        self._fitted = True

    def transform(self, texts):
        scalars = np.array([self._scalar(t) for t in texts], dtype=np.float32)
        if self._fitted:
            clean   = [t if t.strip() else "empty" for t in texts]
            tfidf_m = self.tfidf.transform(clean).toarray().astype(np.float32)
            return np.hstack([scalars, tfidf_m])
        return scalars

    def fit_transform(self, texts):
        self.fit(texts); return self.transform(texts)


# ── AUDIO ─────────────────────────────────────────────────────────────────────

class AudioFeatureExtractor:
    """
    Option A: Uses librosa from raw WAV files.
    Same feature space for training and live inference.
    Feature vector: 144 dims (MFCC + chroma + mel, mean+std)
    """

    def transform_one(self, audio_path) -> np.ndarray:
        return extract_audio_features_from_wav(audio_path)

    def transform(self, audio_paths) -> np.ndarray:
        rows = [self.transform_one(p)
                for p in tqdm(audio_paths, desc="Audio (librosa WAV)")]
        return np.array(rows, dtype=np.float32)

    @property
    def n_features(self): return (N_MFCC + N_CHROMA + N_MEL) * 2  # 144


# ── VIDEO ─────────────────────────────────────────────────────────────────────

class VideoFeatureExtractor:
    """
    Training: uses CLNF_AUs.txt pre-extracted features.
    Live: uses DeepFace on webcam frames (via webcam_features.py).
    Both return 40-dim vectors (20 AUs × mean+std).
    """

    def transform_one(self, clnf_path) -> np.ndarray:
        n_out = len(AU_COLS) * 2
        if not clnf_path or not Path(str(clnf_path)).exists():
            return np.zeros(n_out, dtype=np.float32)
        try:
            df = pd.read_csv(str(clnf_path), sep=",", skipinitialspace=True)
            df.columns = [c.strip() for c in df.columns]
            if "success"    in df.columns: df = df[df["success"] == 1]
            if "confidence" in df.columns: df = df[df["confidence"] > 0.5]
            if len(df) == 0:
                return np.zeros(n_out, dtype=np.float32)
            cols = [c for c in AU_COLS if c in df.columns]
            arr  = df[cols].apply(pd.to_numeric, errors="coerce").fillna(0).values
            if arr.shape[1] < len(AU_COLS):
                pad = np.zeros((arr.shape[0], len(AU_COLS) - arr.shape[1]))
                arr = np.hstack([arr, pad])
            return np.concatenate([arr.mean(axis=0), arr.std(axis=0)]).astype(np.float32)
        except Exception as e:
            print(f"  [WARN] CLNF error {clnf_path}: {e}")
            return np.zeros(len(AU_COLS) * 2, dtype=np.float32)

    def transform(self, clnf_paths) -> np.ndarray:
        rows = [self.transform_one(p) for p in tqdm(clnf_paths, desc="Video (CLNF_AUs)")]
        return np.array(rows, dtype=np.float32)

    @property
    def n_features(self): return len(AU_COLS) * 2  # 40


# ── COMBINED ──────────────────────────────────────────────────────────────────

class MultimodalFeatureExtractor:
    """
    Text (60) + Audio/librosa (144) + Video/CLNF (40) = 244 features
    Training: uses WAV files directly (Option A)
    Live: same audio feature space, DeepFace for video
    """

    def __init__(self):
        self.text_ext  = TextFeatureExtractor()
        self.audio_ext = AudioFeatureExtractor()
        self.video_ext = VideoFeatureExtractor()
        self.scaler    = StandardScaler()
        self._fitted   = False

    def _extract(self, df, fit_text=False):
        from data_loader import load_transcript
        print("\n-- Extracting text features --")
        texts   = [load_transcript(p) for p in df["transcript"].tolist()]
        X_text  = self.text_ext.fit_transform(texts) if fit_text else self.text_ext.transform(texts)

        print("-- Extracting audio features (librosa WAV) --")
        # Option A: use raw WAV files, not COVAREP
        X_audio = self.audio_ext.transform(df["audio"].tolist())

        print("-- Extracting video features (CLNF_AUs) --")
        X_video = self.video_ext.transform(df["clnf_aus"].tolist())

        return np.nan_to_num(np.hstack([X_text, X_audio, X_video])).astype(np.float32)

    def fit_transform(self, df):
        X = self._extract(df, fit_text=True)
        X = self.scaler.fit_transform(X)
        self._fitted = True
        print(f"\n[Features] Matrix: {X.shape} (text={self.text_ext.transform(['test']).shape[1]}"
              f" + audio={self.audio_ext.n_features} + video={self.video_ext.n_features})")
        return X

    def transform(self, df):
        X = self._extract(df, fit_text=False)
        if self._fitted: X = self.scaler.transform(X)
        return X

    def transform_live(self, text="", audio_path=None, frame_paths=None):
        """
        Live inference: same feature space as training.
        audio_path: path to WAV file recorded during conversation
        frame_paths: list of webcam frame image paths
        """
        from webcam_features import extract_deepface_features

        X_text  = self.text_ext.transform([text])
        X_audio = self.audio_ext.transform_one(audio_path)
        X_video = (extract_deepface_features(frame_paths)
                   if frame_paths and len(frame_paths) > 0
                   else np.zeros(self.video_ext.n_features, dtype=np.float32))

        X = np.nan_to_num(np.hstack([
            X_text, X_audio.reshape(1,-1), X_video.reshape(1,-1)
        ])).astype(np.float32)

        if self._fitted:
            X = self.scaler.transform(X)
        return X

    @property
    def n_features(self):
        return 60 + self.audio_ext.n_features + self.video_ext.n_features


if __name__ == "__main__":
    from data_loader import load_dataset
    df  = load_dataset()
    ext = MultimodalFeatureExtractor()
    X   = ext.fit_transform(df)
    print(f"\nFinal feature matrix: {X.shape}")
    print(f"Text: 60, Audio: {ext.audio_ext.n_features}, Video: {ext.video_ext.n_features}")