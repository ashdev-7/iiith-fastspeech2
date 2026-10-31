"""Stage "ingest": HF parquet -> inventory + 22.05 kHz mono wav/.lab files for candidate clips.

For every row the inventory records the original sample rate, channel count, duration,
a SHA-1 of the decoded PCM (for leakage checks), the raw and model-input transcripts and,
for rejected rows, the reason. Clips that pass the filters are written as
<out>/raw_data/<lang>/<speaker>/<utt_id>.wav (+ .lab), the layout the reference
preprocessor and MFA expect. Nothing is selected for training here; see select_split.py.

Usage (Kaggle):
  python -m data.ingest --lang en --cache /kaggle/temp/hf --out /kaggle/working
"""
import argparse
import csv
import hashlib
import io
import json
import os
import re
import sys

import librosa
import numpy as np
import pyarrow.parquet as pq
import soundfile as sf
from huggingface_hub import snapshot_download

DATASETS = {
    "en": {"repo": "ai4bharat/Svarah", "revision": "ebbf7777fe771490696a3f7b007097606fa8c924",
           "audio_col": "audio_filepath"},
    "te": {"repo": "SPRINGLab/IndicTTS_Telugu", "revision": "7ab3c18491bef3f5d4e820c2701b5f78f54031d6",
           "audio_col": "audio"},
}
TARGET_SR = 22050
MAX_WAV_VALUE = 32767.0  # reference uses 32768.0, which wraps the peak sample to -32768 in int16
MIN_DUR, MAX_DUR = 1.0, 15.0
SVARAH_PROFILE = ["gender", "age-group", "primary_language", "native_place_state", "native_place_district"]
SVARAH_NAME_RE = re.compile(r"^(\d+)_f(\d+)_chunk_(\d+)\.wav$")
INVENTORY_FIELDS = ["utt_id", "speaker", "group", "source_path", "orig_sr", "channels", "duration",
                    "pcm_sha1", "raw_text", "model_text", "status", "reason"]


def english_model_text(raw):
    """Reference english_cleaners (ASCII, lowercase, numbers, abbreviations), then keep the
    characters MFA's English dictionary can use: letters, apostrophes and word spaces."""
    from text.cleaners import english_cleaners
    text = english_cleaners(raw.replace("\r", " ").replace("\n", " "))
    text = re.sub(r"[^a-z' ]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def telugu_model_text(raw):
    """Telugu normalisation + G2P validity check; the .lab keeps words (MFA needs words),
    punctuation becomes a word boundary."""
    from text.telugu import G2PError, sentence_to_phones, normalize
    from text.telugu.g2p import PUNCTUATION
    norm = normalize(raw)
    sentence_to_phones(norm, do_normalize=False)  # raises G2PError on anything unconvertible
    norm = "".join(" " if c in PUNCTUATION else c for c in norm)
    return re.sub(r"\s+", " ", norm).strip()


def speaker_and_group(lang, row, path, index):
    """Return (utt_id, speaker, group). See plan.txt R2.3 for the Svarah speaker proxy."""
    if lang == "en":
        profile = "|".join(str(row[k]) for k in SVARAH_PROFILE)
        speaker = "svarah_" + hashlib.sha1(profile.encode("utf-8")).hexdigest()[:8]
        m = SVARAH_NAME_RE.match(os.path.basename(path or ""))
        group = f"{m.group(1)}_f{m.group(2)}" if m else f"row{index}"
        utt_id = os.path.splitext(os.path.basename(path))[0] if m else f"svarah_row{index}"
        return utt_id, speaker, group, profile
    speaker = f"te_{row['gender']}"
    return f"te_{index:05d}", speaker, None, speaker  # Telugu groups are set by text later


def iter_rows(snapshot_dir, audio_col, columns):
    files = sorted(os.path.join(snapshot_dir, "data", f) for f in os.listdir(os.path.join(snapshot_dir, "data"))
                   if f.endswith(".parquet"))
    index = 0
    for path in files:
        pf = pq.ParquetFile(path)
        for batch in pf.iter_batches(batch_size=64, columns=[audio_col] + columns):
            for row in batch.to_pylist():
                yield index, row
                index += 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", choices=sorted(DATASETS), required=True)
    ap.add_argument("--cache", required=True, help="HF download directory (not saved as output)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0, help="process only N rows (smoke test)")
    args = ap.parse_args()

    spec = DATASETS[args.lang]
    snapshot = snapshot_download(spec["repo"], repo_type="dataset", revision=spec["revision"],
                                 allow_patterns=["data/*"], cache_dir=args.cache)
    meta_cols = (["text"] + SVARAH_PROFILE) if args.lang == "en" else ["text", "gender"]
    gender_names = None
    if args.lang == "te":
        features = json.loads(pq.read_schema(os.path.join(snapshot, "data", sorted(
            os.listdir(os.path.join(snapshot, "data")))[0])).metadata[b"huggingface"])["info"]["features"]
        gender_names = features["gender"]["names"]

    wav_root = os.path.join(args.out, "raw_data", args.lang)
    man_dir = os.path.join(args.out, "manifests", args.lang)
    os.makedirs(man_dir, exist_ok=True)
    inv_path = os.path.join(man_dir, "inventory.tsv")
    seen_ids, profiles = set(), {}
    counts = {"rows": 0, "ok": 0}
    with open(inv_path, "w", encoding="utf-8", newline="") as inv:
        writer = csv.DictWriter(inv, fieldnames=INVENTORY_FIELDS, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        for index, row in iter_rows(snapshot, spec["audio_col"], meta_cols):
            if args.limit and index >= args.limit:
                break
            if gender_names is not None:
                row["gender"] = gender_names[row["gender"]]
            audio = row.pop(spec["audio_col"]) or {}
            src_path = audio.get("path") or ""
            utt_id, speaker, group, profile = speaker_and_group(args.lang, row, src_path, index)
            profiles[speaker] = profile
            rec = {"utt_id": utt_id, "speaker": speaker, "group": group or "", "source_path": src_path,
                   "raw_text": (row.get("text") or "").replace("\t", " ").replace("\r", "\\r").replace("\n", "\\n"),
                   "status": "candidate", "reason": "", "model_text": ""}
            counts["rows"] += 1
            try:
                if utt_id in seen_ids:
                    raise ValueError("duplicate utterance id")
                seen_ids.add(utt_id)
                wav, sr = sf.read(io.BytesIO(audio["bytes"]), dtype="float32", always_2d=True)
                rec.update(orig_sr=sr, channels=wav.shape[1], duration=round(len(wav) / sr, 4),
                           pcm_sha1=hashlib.sha1(wav.tobytes()).hexdigest())
                text = row.get("text") or ""
                if not text.strip():
                    raise ValueError("empty transcript")
                rec["model_text"] = english_model_text(text) if args.lang == "en" else telugu_model_text(text)
                if not rec["model_text"]:
                    raise ValueError("transcript empty after normalisation")
                if not MIN_DUR <= rec["duration"] <= MAX_DUR:
                    raise ValueError(f"duration outside [{MIN_DUR}, {MAX_DUR}] s")
                mono = wav.mean(axis=1)
                peak = float(np.max(np.abs(mono)))
                if not np.isfinite(peak) or peak < 1e-4:
                    raise ValueError("silent or non-finite audio")
                mono = librosa.resample(mono, orig_sr=sr, target_sr=TARGET_SR, res_type="soxr_hq") if sr != TARGET_SR else mono
                mono = mono / np.max(np.abs(mono)) * MAX_WAV_VALUE
                spk_dir = os.path.join(wav_root, speaker)
                os.makedirs(spk_dir, exist_ok=True)
                sf.write(os.path.join(spk_dir, utt_id + ".wav"), mono.astype(np.int16), TARGET_SR, subtype="PCM_16")
                with open(os.path.join(spk_dir, utt_id + ".lab"), "w", encoding="utf-8", newline="\n") as f:
                    f.write(rec["model_text"])
                counts["ok"] += 1
            except Exception as e:  # recorded per row; the run continues
                rec["status"], rec["reason"] = "rejected", f"{type(e).__name__}: {e}"[:300]
            writer.writerow({k: rec.get(k, "") for k in INVENTORY_FIELDS})
            if counts["rows"] % 1000 == 0:
                print(f"[{args.lang}] {counts['rows']} rows, {counts['ok']} candidates", flush=True)

    with open(os.path.join(man_dir, "speakers_profile.json"), "w", encoding="utf-8") as f:
        json.dump(profiles, f, ensure_ascii=False, indent=1)
    with open(os.path.join(man_dir, "ingest_info.json"), "w", encoding="utf-8") as f:
        json.dump({**spec, "target_sr": TARGET_SR, "min_dur": MIN_DUR, "max_dur": MAX_DUR, **counts}, f, indent=1)
    print(f"[{args.lang}] done: {counts}", flush=True)


if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    main()
