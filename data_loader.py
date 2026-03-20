import pandas as pd
import numpy as np
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"
PHQ8_THRESHOLD = 10


def load_labels():
    dfs = []
    for csv_name in [
        "train_split_Depression_AVEC2017.csv",
        "dev_split_Depression_AVEC2017.csv",
        "Detailed_PHQ8_Labels.csv"
    ]:
        p = DATA_DIR / csv_name
        if not p.exists():
            continue
        df = pd.read_csv(p)
        if "PHQ_8Total" in df.columns:
            df = df.rename(columns={"PHQ_8Total": "PHQ8_Score"})
        if "PHQ8_Score" not in df.columns:
            continue
        if "PHQ8_Binary" not in df.columns:
            df["PHQ8_Binary"] = (df["PHQ8_Score"] >= PHQ8_THRESHOLD).astype(int)
        dfs.append(df[["Participant_ID", "PHQ8_Score", "PHQ8_Binary"]])

    if not dfs:
        raise FileNotFoundError(f"No label CSVs found in {DATA_DIR}")

    labels = pd.concat(dfs, ignore_index=True)
    labels = labels.drop_duplicates(subset="Participant_ID", keep="first")
    labels = labels.rename(columns={"PHQ8_Binary": "label"})
    labels["Participant_ID"] = labels["Participant_ID"].astype(int)
    labels["label"] = labels["label"].astype(int)
    print(f"[Labels] {len(labels)} total | Depressed: {labels['label'].sum()} | Not: {(labels['label']==0).sum()}")
    return labels.reset_index(drop=True)


def get_participant_files(pid):
    d = DATA_DIR / f"{pid}_P"
    return {
        "transcript": d / f"{pid}_TRANSCRIPT.csv",
        "covarep":    d / f"{pid}_COVAREP.csv",
        "clnf_aus":   d / f"{pid}_CLNF_AUs.txt",
        "audio":      d / f"{pid}_AUDIO.wav",
        "formant":    d / f"{pid}_FORMANT.csv",
    }


def load_transcript(path):
    if not path or not Path(str(path)).exists():
        return ""
    try:
        df = pd.read_csv(str(path), sep="\t", on_bad_lines="skip")
        if "speaker" in df.columns and "value" in df.columns:
            df = df[df["speaker"].str.strip().str.upper() == "PARTICIPANT"]
            return " ".join(df["value"].dropna().astype(str).tolist()).strip()
    except Exception as e:
        print(f"  [WARN] Transcript error {path}: {e}")
    return ""


def load_dataset():
    labels = load_labels()
    rows = []
    for _, row in labels.iterrows():
        pid = int(row["Participant_ID"])
        if not (DATA_DIR / f"{pid}_P").exists():
            continue
        f = get_participant_files(pid)
        rows.append({
            "participant_id": pid,
            "phq8_score":     float(row["PHQ8_Score"]),
            "label":          int(row["label"]),
            "transcript":     f["transcript"],
            "covarep":        f["covarep"],
            "clnf_aus":       f["clnf_aus"],
            "audio":          f["audio"],
            "formant":        f["formant"],
        })
    df = pd.DataFrame(rows).reset_index(drop=True)
    print(f"[Dataset] {len(df)} participants | Depressed: {df['label'].sum()} | Not: {(df['label']==0).sum()}")
    return df


if __name__ == "__main__":
    df = load_dataset()
    print(df[["participant_id", "phq8_score", "label"]].to_string())
