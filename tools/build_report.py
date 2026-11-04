"""Build Report/Technical_Report.pdf from Report/src/report.html.

Fills the data-driven placeholders from the files in the repository (Results/, Evaluation/), so every
number in those parts of the report is traceable to an output file, then prints the page to PDF with
headless Microsoft Edge (Chrome works too: pass --browser).

  python tools/build_report.py [--browser "C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe"]
"""
import argparse
import html
import json
import os
import subprocess

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "Report", "src", "report.html")
BUILT = os.path.join(ROOT, "Report", "src", "report.built.html")
PDF = os.path.join(ROOT, "Report", "Technical_Report.pdf")

# Per-sentence analysis of the variance plots. Written from the per-word values in
# Results/<Lang>/variance_raw/<ID>.json (word duration = sum of its phones; word pitch = max phone F0).
ANALYSIS = {
    "E1": [
        "<i>antidisestablishmentarianism</i> takes 1,962 ms, 31 % of the 6.32 s sentence, with uniformly short phones and no pause before or after it.",
        "Pitch has several similar peaks (<i>considerable</i> 250 Hz, <i>antidisestablishmentarianism</i> 243, <i>movement</i> 239, <i>momentum</i> 232 Hz) and falls to its minimum on the last word <i>proceedings</i> (199 Hz): a flat, list-like contour with final declination (first third 216 → last third 200 Hz).",
        "<i>proceedings</i> is the longest word after the long word (929 ms) and its final vowel is lengthened 1.3×, a phrase-final cue. No pause is predicted, because the sentence has no internal punctuation.",
    ],
    "E2a": [
        "<i>conduct</i> (noun, <code>K AA1 N D AH0 K T</code>): first syllable 313 ms vs second 325 ms (ratio 0.96). That is relatively more weight on the first syllable than the verb in E2b gets (0.82), but not an absolute first-syllable dominance. Pitch rises across the word (199 → 222 Hz on its vowels), so in this voice the noun stress is cued weakly, by relative duration only.",
        "Main accent on <i>orchestra's</i> (241 Hz) and <i>ceremony</i> (234 Hz), then a fall to 190 Hz on <i>audience</i>.",
        "The final vowel is short (0.7× the mean) because <i>audience</i> ends in /s/; the word as a whole is long (615 ms). <i>conduct</i> as a noun is 639 ms, the longest word of the sentence.",
    ],
    "E2b": [
        "<i>conduct</i> (verb, <code>K AA0 N D AH1 K T</code>): first syllable 209 ms vs second 255 ms (ratio 0.82, shifted towards the second syllable relative to E2a's 0.96), with the higher pitch on the stressed second vowel (AH1 215 Hz vs AA0 209 Hz).",
        "The word is 464 ms against 639 ms as a noun: the verb is less prominent in its phrase than the subject noun of E2a. The homograph contrast is carried mainly by duration.",
        "Peaks on <i>evaluate</i> (240 Hz) and <i>researchers</i> (235 Hz); declination to <i>method</i> (202 Hz) with 1.5× final lengthening. <i>to evaluate</i> is fast (128 ms for <i>to</i>), where the ASR heard \"experimental evaluation\".",
    ],
    "E3": [
        "The split word is pronounced as <i>rhythm</i> (<code>R IH1 DH AH0 M</code>, 383 ms) with the sentence's highest pitch (270 Hz, and 261 Hz on <i>of</i>): a strong initial accent.",
        "<i>changed</i> is low (213 Hz); <i>unexpectedly</i> is very long (1,115 ms), with a final vowel of 244 ms (2.9× the mean vowel), the strongest final lengthening of the ten sentences.",
        "No pause is predicted between <i>rhythm</i> and <i>of</i>; the double space in the text has no effect after normalisation.",
    ],
    "E4": [
        "Numbers are read as words with growing durations (<i>ten</i> 232, <i>forty</i> 325, <i>five</i> 383 ms; <i>twelve</i> 453, <i>November</i> 476, <i>six</i> 418 ms): list-like, each number group ends long, while <i>twenty twenty</i> is fast (244 + 221 ms). <i>AM</i> takes 453 ms with the letter-name duration rule (244 ms without it).",
        "The comma yields a <b>546 ms pause</b> (7.6 % of the sentence; 824 ms with the former voice, so the speaker embedding changes pause length). After it there is no pitch reset (<i>six</i> 215 → <i>so</i> 219 Hz), and the contour declines to the sentence minimum on <i>time</i> (194 Hz); the highest pitch is on <i>meeting</i> (267 Hz).",
        "<i>andaroo</i> (<code>AE0 N D AA1 R UW0</code>) gets 557 ms with stress on the second syllable; final lengthening 1.8× on <i>time</i>.",
    ],
    "T1": [
        "<span class=\"te\">న్యూరోఫార్మకోలాజికల్</span> lasts 1,695 ms (32 % of 5.27 s) and carries the sentence's highest pitch (313 Hz): the long loanword is the focus of the sentence-initial phrase.",
        "Pitch drops to 230 Hz on <span class=\"te\">పరిశోధనలో</span> and stays in a narrow 237–258 Hz band for the rest: a typical Telugu (verb-final) declarative pattern with the main prominence early.",
        "The final verb <span class=\"te\">విశ్లేషించారు</span> is long (964 ms; cluster <code>sh l</code>, nasal <code>ny</code> before <code>c</code>) with 1.3× final lengthening; no internal pauses (no internal punctuation).",
    ],
    "T2a": [
        "Peak pitch on the first word <span class=\"te\">తాడుపై</span> (326 Hz), then a steady fall to <span class=\"te\">ఉంది</span> (207 Hz).",
        "<span class=\"te\">పట్టు</span> (<code>p a tt tt u</code>, geminate retroflex) lasts 337 ms with 242 Hz — inside the falling line, not accented.",
        "Final vowel 151 ms, 1.8× the mean: strong phrase-final lengthening.",
    ],
    "T2b": [
        "<span class=\"te\">పట్టు</span> has exactly the same phones and the same duration (337 ms) as in T2a, as it should (the two words are homographs and homophones), but now carries the sentence's pitch peak (295 Hz vs 242 Hz in T2a): as the modifier of the compound <span class=\"te\">పట్టుచీర</span> (silk saree) it is accented, after a low start on <span class=\"te\">ఆమె</span> (220 Hz).",
        "The rest declines from 254 Hz (<span class=\"te\">చీరను</span>) to 202 Hz (<span class=\"te\">వచ్చింది</span>), with 1.7× final lengthening.",
        "The difference between T2a and T2b is therefore purely prosodic (position and focus), which the model reproduces.",
    ],
    "T3": [
        "Peak on <span class=\"te\">పులి</span> (321 Hz, the topic).",
        "The broken word <span class=\"te\">అడ  విలో</span> (for <span class=\"te\">అడవిలో</span>) gives exactly the phone sequence of <span class=\"te\">అడవిలో</span> (<code>a dd a v i l oo</code>; checked), because the Telugu input has no word-boundary symbol: the model receives the intended word. It is spoken without a pause (<span class=\"te\">అడ</span> 186 ms, <span class=\"te\">విలో</span> 279 ms, contiguous), with pitch rising from 217 to 274 Hz across the word.",
        "<span class=\"te\">నిశ్శబ్దంగా</span> is long (778 ms; geminate <code>sh sh</code> and the cluster <code>b d</code>); <span class=\"te\">నడిచింది</span> ends the sentence at the pitch minimum (194 Hz) with 2.1× final lengthening.",
    ],
    "T4": [
        "Longest sentence (9.13 s, 120 phones). Numbers are expanded to Telugu words: <span class=\"te\">రెండు వేల ఇరవై ఆరు</span> (302/290/453/267 ms), <span class=\"te\">పన్నెండవ</span> (615 ms), <span class=\"te\">పది నలభై ఐదు</span> (244/418/267 ms), read without hesitation pauses.",
        "The comma gives a 546 ms pause followed by a clear <b>pitch reset</b>: from ≈ 208 Hz on <span class=\"te\">ప్రారంభమవుతుంది</span> (the longest word, 1,126 ms) up to 270 Hz on <span class=\"te\">సమయానికి</span> — the second clause starts a new intonation phrase.",
        "Highest pitch on <span class=\"te\">సమావేశం</span> (272 Hz), lowest on the final <span class=\"te\">ఉంచండి</span> (198 Hz), with 1.5× final lengthening.",
    ],
}


def pct(x):
    return "{:.1f} %".format(100 * x)


def sentence_figs(lang_dir, ids, texts):
    out = []
    for i in ids:
        cls = ' class="te"' if lang_dir == "Telugu" else ""
        bullets = "".join("<li>{}</li>".format(b) for b in ANALYSIS[i])
        imgs = "".join('<img src="../../Results/{0}/variance_adaptor_plots/{1}/{2}_{1}.png">'.format(lang_dir, k, i)
                       for k in ("duration", "pitch", "energy"))
        out.append('<div class="sent"><h4>{} — <span{}>{}</span></h4><ul>{}</ul><div class="plots">{}</div></div>'.format(
            i, cls, html.escape(texts[i]), bullets, imgs))
    return "\n".join(out)


def main(browser):
    s = open(SRC, encoding="utf-8").read()
    tests = {it["id"]: it["canonical"] for it in json.load(open(os.path.join(ROOT, "tests", "test_sentences.json"),
                                                                 encoding="utf-8"))["items"]}
    te = json.load(open(os.path.join(ROOT, "Evaluation", "Telugu", "metrics.json"), encoding="utf-8"))
    en_s = json.load(open(os.path.join(ROOT, "Evaluation", "English", "metrics_sentences.json"), encoding="utf-8"))
    stress = {it["id"]: it for it in json.load(open(os.path.join(ROOT, "Evaluation", "stress_sentences.json"),
                                                    encoding="utf-8"))["items"]}
    te_sent = {r["id"]: r for r in te["sentences"]}
    en_sent = {r["id"]: r for r in en_s["sentences"]}
    intel = te.get("intelligibility", {})
    if "synthesis" not in intel:
        raise SystemExit("Evaluation/Telugu/metrics.json has no intelligibility results: " + json.dumps(intel)[:300])

    syn, cop = intel["synthesis"], intel["copy_synthesis_ceiling"]
    rep = {
        "TE_CER_GLANCE": "{} ({})".format(pct(syn["cer"]), pct(cop["cer"])),
        "TE_SYN_CER": "{} / {}".format(pct(syn["cer"]), pct(syn["wer"])),
        "TE_COPY_ROW": "{} / {}".format(pct(cop["cer"]), pct(cop["wer"])),
        "TE_COPY_CER": pct(cop["cer"]),
    }
    rows = []
    for i in ("T1", "T2a", "T2b", "T3", "T4"):
        r = te_sent[i]
        notes = {"T1": "all words by rule; <code>ny</code> before <code>c</code> in విశ్లేషించారు",
                 "T2a": "పట్టు → <code>p a tt tt u</code>",
                 "T2b": "పట్టు → <code>p a tt tt u</code> (same as T2a)",
                 "T3": "the split అడ  విలో kept as two words; no pause between them",
                 "T4": "2026 → రెండు వేల ఇరవై ఆరు; 12వ → పన్నెండవ; 10:45 → పది నలభై ఐదు"}[i]
        rows.append('<tr><td>{}</td><td class="n">{:.2f} s</td><td class="te">{}</td><td class="te">{}</td><td class="n">{}</td></tr>'.format(
            i, r["audio_s"], notes, html.escape(r.get("asr_transcript", "—")),
            pct(r["asr_vs_model_input"]["cer"]) if r.get("asr_vs_model_input") else "—"))
    rep["TE_SENTENCE_ROWS"] = "\n".join(rows)
    cers = {i: te_sent[i]["asr_vs_model_input"]["cer"] for i in ("T1", "T2a", "T2b", "T3", "T4")}
    best, worst = min(cers, key=cers.get), max(cers, key=cers.get)
    rep["TE_OBSERVATIONS"] = (
        "All five sentences are synthesised without errors in the frontend, with every number expanded to Telugu words. "
        "Against Whisper's Telugu recognition the clearest sentence is {} (CER {}) and the hardest {} (CER {}); "
        "the mean over the five is {}. Transcripts should be read with Whisper's weak Telugu ability in mind (below). "
        "What the transcripts show, sentence by sentence: "
        "<b>T1</b> — Whisper recognises the long English loanword and writes it in Latin script (<i>Neuro-Pharmacological</i>), "
        "so most of T1's CER is a script mismatch, not a mispronunciation; the remaining errors are శ → స in "
        "పరిశోధనలో and విశ్లేషించారు and ఔ → ఉ in ఔషధాల. "
        "<b>T2a/T2b</b> — పట్టు is transcribed correctly in both senses (the two are written and pronounced identically; "
        "only the pitch differs, §6.2); the errors are word-boundary merges (పయతని, ఆమేపట్టు) and vowel length "
        "(ఆమె → ఆమే, చీర → చిర). "
        "<b>T3</b> — the split అడ  విలో is heard as the single word అడవిలో, i.e. the model joins it as a listener would; "
        "the only error is again శ → స (నిస్సబ్దంగా). "
        "<b>T4</b> — Whisper writes the expanded numbers back as digits (2026, 10.45), which the CER counts as errors, "
        "drops the opening ఈ సమావేశం, and merges నవంబర్ పన్నెండవ తేదీ across word boundaries (the syllables are all "
        "present); ప్రారంభ → ప్రారంబ is a lost aspiration. "
        "The one systematic pattern is శ (frontend phone <code>sh</code>) heard as స (<code>s</code>) in all four words "
        "that contain it. The frontend keeps the two phones distinct, so the merger is in the acoustic model or in "
        "Whisper; with only these transcripts the two cannot be separated <span class=\"tag est\">estimated</span>.").format(
        best, pct(cers[best]), worst, pct(cers[worst]), pct(sum(cers.values()) / 5))
    rep["TE_FAILURES"] = ("శ heard as స in every word that has it; vowel-length and word-boundary errors; "
                          "highest ASR error on {} (CER {}, partly digits written by Whisper); see §5.2").format(
                              worst, pct(cers[worst]))
    rep["TE_EN_SENTENCE_FIGS"] = ("<h3>6.1 English</h3>\n" + sentence_figs("English", ["E1", "E2a", "E2b", "E3", "E4"], tests)
                                  + '\n<h3 class="pb">6.2 Telugu</h3>\n'
                                  + sentence_figs("Telugu", ["T1", "T2a", "T2b", "T3", "T4"], tests))

    def stress_table(ids, sent, lang):
        cls = ' class="te"' if lang == "te" else ""
        out = ['<table><tr><th>ID</th><th>Case</th><th>Text</th><th>Whisper transcript</th><th class="n">CER</th></tr>']
        for i in ids:
            r = sent[i]
            m = r.get("asr_vs_original_whisper_norm") or r.get("asr_vs_model_input")
            text = html.escape(stress[i]["text"]) + (
                '<br><span class="small">{}</span>'.format(html.escape(stress[i]["gloss"])) if stress[i].get("gloss") else "")
            out.append('<tr><td>{}</td><td>{}</td><td{}>{}</td><td{}>{}</td><td class="n">{}</td></tr>'.format(
                i, html.escape(stress[i]["category"]), cls, text, cls, html.escape(r.get("asr_transcript", "—")),
                pct(m["cer"]) if m else "—"))
        out.append("</table>")
        return "\n".join(out)

    rep["STRESS_TABLES"] = (
        "<h4>English (CER after Whisper's English normaliser, against the original text)</h4>"
        + stress_table(["SE0%d" % k for k in range(1, 9)], en_sent, "en")
        + "<h4>Telugu (CER against the normalised text)</h4>"
        + stress_table(["ST0%d" % k for k in range(1, 9)], te_sent, "te"))

    for k, v in rep.items():
        if k not in s:
            raise SystemExit("placeholder {} missing from report.html".format(k))
        s = s.replace(k, v)
    with open(BUILT, "w", encoding="utf-8") as f:
        f.write(s)
    url = "file:///" + BUILT.replace("\\", "/")
    subprocess.run([browser, "--headless=new", "--disable-gpu", "--no-pdf-header-footer",
                    "--print-to-pdf=" + PDF, url], check=True, timeout=300)
    print("wrote", PDF, os.path.getsize(PDF), "bytes")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--browser", default=r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe")
    main(ap.parse_args().browser)
