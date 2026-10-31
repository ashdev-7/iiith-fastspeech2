"""Stage "align" (preparation): MFA corpus of the selected utterances + pronunciation dictionary.

Only train and val utterances are aligned (test sentences are never seen). The corpus is a
tree of symlinks <corpus>/<speaker>/<utt>.wav|.lab into the ingest output.

Dictionaries (word<TAB>phones):
  en: reference lexicon (lexicon/librispeech-lexicon.txt, ARPAbet with stress, the phone set of
      MFA's english_us_arpa model); words missing from it get g2p_en pronunciations, the same
      fallback the inference frontend uses. Every pronunciation variant of a word is kept so MFA
      can choose (e.g. CONDUCT noun/verb).
  te: rule-based G2P (text/telugu), one pronunciation per word.

Usage:
  python data/mfa_prepare.py --lang te --ingest_dir /kaggle/input/... --corpus /kaggle/temp/corpus_te \
      --dict_out /kaggle/working/mfa/te/dictionary.txt
"""
import argparse
import csv
import json
import os
import re
import sys
from collections import Counter, defaultdict

LEXICON = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "lexicon",
                       "librispeech-lexicon.txt")


def selected_utterances(manifest_dir):
    utts = []
    for split in ("train", "val"):
        with open(os.path.join(manifest_dir, f"{split}.tsv"), encoding="utf-8", newline="") as f:
            utts += [dict(r, split=split) for r in csv.DictReader(f, delimiter="\t")]
    return utts


def read_reference_lexicon():
    lex = defaultdict(list)
    with open(LEXICON, encoding="utf-8") as f:
        for line in f:
            parts = re.split(r"\s+", line.strip())
            if len(parts) >= 2 and parts[1:] not in lex[parts[0].lower()]:
                lex[parts[0].lower()].append(parts[1:])
    return lex


def english_dictionary(words):
    lex = read_reference_lexicon()
    oov = sorted(w for w in words if w not in lex)
    from text.english_frontend import _get_g2p  # also fetches the nltk resources g2p_en needs
    g2p = _get_g2p()
    entries, g2p_words = {}, {}
    for w in sorted(words):
        if w in lex:
            entries[w] = lex[w]
        else:
            phones = [p for p in g2p(w) if p.strip() and re.match(r"^[A-Z]+[012]?$", p)]
            if phones:
                entries[w] = [phones]
                g2p_words[w] = " ".join(phones)
    return entries, {"oov_words": len(oov), "g2p_words": g2p_words}


def telugu_dictionary(words):
    from text.telugu import word_to_phones
    return {w: [word_to_phones(w)] for w in sorted(words)}, {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", choices=["en", "te"], required=True)
    ap.add_argument("--ingest_dir", required=True, help="output folder of the ingest job")
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--dict_out", required=True)
    args = ap.parse_args()

    utts = selected_utterances(os.path.join(args.ingest_dir, "manifests", args.lang))
    words = Counter()
    for u in utts:
        src = os.path.join(args.ingest_dir, "raw_data", args.lang, u["speaker"], u["utt_id"])
        dst_dir = os.path.join(args.corpus, u["speaker"])
        os.makedirs(dst_dir, exist_ok=True)
        for ext in (".wav", ".lab"):
            if not os.path.exists(src + ext):
                raise SystemExit(f"missing {src + ext}")
            dst = os.path.join(dst_dir, u["utt_id"] + ext)
            if not os.path.exists(dst):
                os.symlink(src + ext, dst)
        with open(src + ".lab", encoding="utf-8") as f:
            words.update(f.read().split())

    build = english_dictionary if args.lang == "en" else telugu_dictionary
    entries, info = build(set(words))
    os.makedirs(os.path.dirname(args.dict_out), exist_ok=True)
    with open(args.dict_out, "w", encoding="utf-8", newline="\n") as f:
        for w in sorted(entries):
            for pron in entries[w]:
                f.write(f"{w}\t{' '.join(pron)}\n")
    missing = sorted(set(words) - set(entries))
    info.update({"utterances": len(utts), "distinct_words": len(words), "dictionary_words": len(entries),
                 "words_without_pronunciation": missing,
                 "phones": sorted({p for prons in entries.values() for pron in prons for p in pron})})
    with open(os.path.splitext(args.dict_out)[0] + "_info.json", "w", encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False, indent=1)
    print(json.dumps({k: (v if not isinstance(v, (list, dict)) else len(v)) for k, v in info.items()}))


if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    main()
