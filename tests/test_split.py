"""select_split: deterministic, quota-respecting, and no train/val overlap."""
import copy

from data.select_split import leakage_report, select


def _rows():
    rows = []
    # Two speakers reading many of the same sentences (as in IndicTTS), 6 s each.
    for i in range(400):
        for spk in ("te_female", "te_male"):
            rows.append({"utt_id": f"{spk}_{i}", "speaker": spk, "group": "", "duration": "6.0",
                         "model_text": f"sentence {i}", "pcm_sha1": f"{spk}{i}", "status": "candidate",
                         "reason": ""})
    rows.append({"utt_id": "dup", "speaker": "te_male", "group": "", "duration": "6.0",
                 "model_text": "other", "pcm_sha1": "te_male0", "status": "candidate", "reason": ""})
    rows.append({"utt_id": "bad", "speaker": "te_male", "group": "", "duration": "0.5",
                 "model_text": "", "pcm_sha1": "x", "status": "rejected", "reason": "ValueError: short"})
    return rows


def test_split_is_deterministic_and_disjoint():
    a, excluded = select(copy.deepcopy(_rows()), train_hours=0.5, val_hours=0.1, seed=1234)
    b, _ = select(copy.deepcopy(_rows()), train_hours=0.5, val_hours=0.1, seed=1234)
    assert [(r["utt_id"], r["split"]) for r in a] == [(r["utt_id"], r["split"]) for r in b]
    assert excluded == {"dup": "duplicate audio of te_male_0"}
    assert leakage_report(a) == {"utt_id_overlap": 0, "group_overlap": 0, "model_text_overlap": 0}


def test_same_sentence_stays_in_one_split():
    rows, _ = select(copy.deepcopy(_rows()), train_hours=0.5, val_hours=0.1, seed=7)
    split_of = {}
    for r in rows:
        assert split_of.setdefault(r["model_text"], r["split"]) == r["split"]


def test_equal_balance_with_unequal_speakers():
    """Mostly distinct sentences per speaker (as in IndicTTS Telugu, where only 93 transcripts
    repeat); the male speaker has 3x more candidate audio."""
    rows = [{"utt_id": f"{spk}_{i}", "speaker": spk, "group": "", "duration": "6.0",
             "model_text": f"{spk} sentence {i}", "pcm_sha1": f"{spk}{i}", "status": "candidate", "reason": ""}
            for spk, n in (("te_female", 400), ("te_male", 1200)) for i in range(n)]
    out, _ = select(copy.deepcopy(rows), train_hours=0.5, val_hours=0.1, seed=3, balance="equal")
    for spk in ("te_female", "te_male"):
        got = sum(r["duration"] for r in out if r["split"] == "train" and r["speaker"] == spk) / 3600
        assert 0.25 * 0.9 <= got <= 0.25 * 1.02 + 1e-9


def test_quotas_respected_per_speaker():
    rows, _ = select(copy.deepcopy(_rows()), train_hours=0.5, val_hours=0.1, seed=1234)
    for split, hours in (("train", 0.5), ("val", 0.1)):
        for spk in ("te_female", "te_male"):
            got = sum(r["duration"] for r in rows if r["split"] == split and r["speaker"] == spk) / 3600
            assert hours / 2 * 0.9 <= got <= hours / 2 * 1.02 + 1e-9


def test_recordings_sharing_a_sentence_never_cross_the_split():
    """Svarah: different recordings (groups) can contain the same prompt; the 2026-10-31 ingest found
    49 such transcripts in both train and val before groups sharing text were merged."""
    rows = []
    for g in range(300):
        for c in range(3):
            text = "can you name five states" if c == 0 and g % 10 == 0 else f"recording {g} chunk {c}"
            rows.append({"utt_id": f"{g}_{c}", "speaker": f"spk{g % 7}", "group": f"rec{g}", "duration": "5.0",
                         "model_text": text, "pcm_sha1": f"{g}_{c}", "status": "candidate", "reason": ""})
    out, _ = select(copy.deepcopy(rows), train_hours=0.6, val_hours=0.15, seed=11)
    assert leakage_report(out)["model_text_overlap"] == 0
    shared = {r["split"] for r in out if r["model_text"] == "can you name five states"}
    assert len(shared) == 1
    assert len({r["group"] for r in out if r["model_text"] == "can you name five states"}) == 1
