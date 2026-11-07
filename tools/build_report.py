"""Build Report/Technical_Report.pdf from Report/src/report.html.

Fills the data-driven placeholders from files in the repository (Results/, Evaluation/), so every number in
those parts of the report is traceable to an output file, refuses to build if a placeholder is left or a
source is missing, then prints the page to PDF with headless Microsoft Edge (Chrome works too: --browser).

  python tools/build_report.py [--browser "C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe"]
"""
import argparse
import html
import json
import os
import re
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
SRC = os.path.join(ROOT, "Report", "src", "report.html")
BUILT = os.path.join(ROOT, "Report", "src", "report.built.html")
PDF = os.path.join(ROOT, "Report", "Technical_Report.pdf")
EV = os.path.join(ROOT, "Evaluation")

from report_text import ANALYSIS, TEXT  # noqa: E402  (hand-written analysis, from the files named in each entry)
sys.path.insert(0, os.path.join(ROOT, "fastspeech2"))
import eval_metrics as M  # noqa: E402


def kannada_to_telugu(text):
    """Whisper large-v3 sometimes writes Telugu speech in Kannada script. The two Unicode blocks have the same
    layout (Kannada U+0C80-0CFF = Telugu U+0C00-0C7F + 0x80), so the transcript is mapped back character by
    character before scoring; the raw CER is reported next to it."""
    return "".join(chr(ord(c) - 0x80) if 0x0C80 <= ord(c) <= 0x0CFF else c for c in text)


def load(*parts):
    path = os.path.join(ROOT, *parts)
    if not os.path.exists(path):
        raise SystemExit("missing source for the report: " + os.path.relpath(path, ROOT))
    return json.load(open(path, encoding="utf-8"))


def pct(x, d=1):
    return ("{:." + str(d) + "f} %").format(100 * x)


def corpus(rows, key):
    rows = [r[key] for r in rows if r.get(key)]
    ce, rc = sum(r["char_edits"] for r in rows), sum(r["ref_chars"] for r in rows)
    return ce / max(1, rc)


def _tall(path):
    """Plots of long sentences are drawn on several rows; they are printed at full width so the labels stay legible."""
    from PIL import Image
    w, h = Image.open(path).size
    return ' class="tall"' if h > 0.5 * w else ""


def sentence_figs(lang_dir, ids, texts):
    out = []
    for i in ids:
        cls = ' class="te"' if lang_dir == "Telugu" else ""
        bullets = "".join("<li>{}</li>".format(b) for b in ANALYSIS[i])
        imgs = "".join('<img{3} src="../../Results/{0}/variance_adaptor_plots/{1}/{2}_{1}.png">'.format(
            lang_dir, k, i, _tall(os.path.join(ROOT, "Results", lang_dir, "variance_adaptor_plots", k, "{}_{}.png".format(i, k))))
                       for k in ("duration", "pitch", "energy"))
        out.append('<div class="sent"><h4>{}: <span{}><span class="ws">{}</span></span></h4><ul>{}</ul>'
                   '<figure class="plots" data-cap="{} variance-adaptor plots: duration, pitch and energy">{}'
                   '<figcaption>From top to bottom: predicted duration per phone; pitch per phone with the F0 of the generated '
                   'audio; energy per phone.</figcaption></figure></div>'.format(
                       i, cls, html.escape(texts[i]), bullets, i, imgs))
    return "\n".join(out)


def main(browser):
    s = open(SRC, encoding="utf-8").read()
    tests = {it["id"]: it["canonical"] for it in load("tests", "test_sentences.json")["items"]}
    stress = {it["id"]: it for it in load("Evaluation", "stress_sentences.json")["items"]}
    ext = {it["id"]: it for it in load("Evaluation", "Extended", "extended_sentences.json")["items"]}
    texts = dict(tests)
    for d in (stress, ext):
        texts.update({k: v.get("canonical") or v.get("text") for k, v in d.items()})

    en = load("Evaluation", "English", "metrics_sentences.json")        # final English configuration
    en_s = {r["id"]: r for r in en["sentences"]}
    te_w = load("Evaluation", "Telugu", "metrics.json")                 # Whisper large-v3, Telugu as submitted
    te_ws = {r["id"]: r for r in te_w["sentences"]}
    for extra in load("Evaluation", "Telugu", "metrics_extended_whisper.json")["sentences"]:  # job 14: XT01-XT10
        te_ws.setdefault(extra["id"], extra)
    for r in te_ws.values():  # Whisper large-v3, with the Kannada-script transcripts mapped to Telugu
        h = r.get("asr_transcript") or ""
        r["asr_script_mapped"] = M.error_rates(r["model_input"], kannada_to_telugu(h))
        r["kannada_chars"] = sum(0x0C80 <= ord(c) <= 0x0CFF for c in h)
    have_te_asr = os.path.exists(os.path.join(EV, "Telugu", "asr_telugu", "metrics_sentences.json"))
    mode = "te_asr" if have_te_asr else "whisper"
    if have_te_asr:
        te_as = {r["id"]: r for r in load("Evaluation", "Telugu", "asr_telugu", "metrics_sentences.json")["sentences"]}
        te_av = load("Evaluation", "Telugu", "asr_telugu", "metrics_validation.json")
    else:
        te_as, te_av = {k: dict(v, asr_vs_model_input=v["asr_script_mapped"],
                                asr_transcript=kannada_to_telugu(v.get("asr_transcript") or "")
                                + (" [Kannada script, mapped]" if v["kannada_chars"] else ""))
                        for k, v in te_ws.items()}, None
    print("Telugu ASR mode:", mode)
    lev = load("Evaluation", "English", "clarity_levels.json")["mean"]
    rep = {}

    # ------------------------------------------------------------------ glance / Section 8
    tests_en = ["E1", "E2a", "E2b", "E3", "E4"]
    tests_te = ["T1", "T2a", "T2b", "T3", "T4"]
    rep["EN_TEST_GLANCE"] = "{}".format(
        pct(corpus([en_s[i] for i in tests_en], "asr_vs_original_whisper_norm")))
    wi = te_w["intelligibility"]
    if "synthesis" not in wi:
        raise SystemExit("Telugu intelligibility results missing")
    rep["TE_SYN_CER"] = "{} / {}".format(pct(wi["synthesis"]["cer"]), pct(wi["synthesis"]["wer"]))
    rep["TE_COPY_ROW"] = "{} / {}".format(pct(wi["copy_synthesis_ceiling"]["cer"]), pct(wi["copy_synthesis_ceiling"]["wer"]))
    w_glance = "{} vs {}".format(pct(wi["synthesis"]["cer"]), pct(wi["copy_synthesis_ceiling"]["cer"]))
    if mode == "te_asr":
        ai = te_av["intelligibility"]
        if "synthesis" not in ai:
            raise SystemExit("Telugu-ASR validation results missing")
        rep["TE_TEST_GLANCE"] = "{} (Telugu ASR)".format(pct(corpus([te_as[i] for i in tests_te], "asr_vs_model_input")))
        rep["TE_VAL_ASR_GLANCE"] = "{} vs {} (Telugu ASR, {} utt.); Whisper: {}".format(
            pct(ai["synthesis"]["cer"]), pct(ai["copy_synthesis_ceiling"]["cer"]), ai["utterances"], w_glance)
        rep["TE2_ROWS"] = ('<tr><td>Telugu ASR CER / WER, our synthesis ({n} utt.)</td><td class="n">n/a</td><td class="n">{s}</td></tr>'
                           '<tr><td>Same, copy-synthesis ceiling</td><td class="n">n/a</td><td class="n">{c}</td></tr>').format(
            n=ai["utterances"], s="{} / {}".format(pct(ai["synthesis"]["cer"]), pct(ai["synthesis"]["wer"])),
            c="{} / {}".format(pct(ai["copy_synthesis_ceiling"]["cer"]), pct(ai["copy_synthesis_ceiling"]["wer"])))
        rep["TE_TABLE_HEAD"] = ('<tr><th>ID</th><th class="n">Duration</th><th>Telugu-ASR transcript of the generated '
                                'audio</th><th class="n">CER, Telugu ASR</th><th class="n">CER, Whisper</th></tr>')
    else:
        rep["TE_TEST_GLANCE"] = "{}".format(
            pct(corpus([te_as[i] for i in tests_te], "asr_vs_model_input")))
        rep["TE_VAL_ASR_GLANCE"] = w_glance
        rep["TE2_ROWS"] = ""
        rep["TE_TABLE_HEAD"] = ('<tr><th>ID</th><th class="n">Duration</th><th>Whisper large-v3 transcript of the '
                                'generated audio</th><th class="n">CER</th><th class="n">CER, script-mapped</th></tr>')
    for k in ("TE_ASR_BOX", "TE_ASR_NOTE"):
        rep[k] = TEXT[k][mode]

    # ------------------------------------------------------------------ Section 6.4 levels
    b, a = lev["before"], lev["after"]
    rep.update({"CLARITY_END_BEFORE": "{:.0f} dBFS".format(b["end_db"]), "CLARITY_END_AFTER": "{:.0f} dBFS".format(a["end_db"]),
                "CLARITY_QUIET_BEFORE": "{:.1f}".format(b["quiet_db"]), "CLARITY_QUIET_AFTER": "{:.1f} dBFS".format(a["quiet_db"]),
                "CLARITY_SPEECH_BEFORE": "{:.1f}".format(b["speech_db"]),
                "CLARITY_SPEECH_AFTER": "{:.1f} dBFS".format(a["speech_db"])})

    # ------------------------------------------------------------------ Section 6.5 English
    rows = []
    for i in tests_en:
        r = en_s[i]
        rows.append('<tr><td>{}</td><td class="n">{:.2f} s</td><td>{}</td><td class="n">{}</td><td class="n">{}</td></tr>'.format(
            i, r["audio_s"], html.escape(r.get("asr_transcript", "n/a")),
            pct(r["asr_vs_model_input"]["cer"]), pct(r["asr_vs_original_whisper_norm"]["cer"])))
    rep["EN_SENTENCE_ROWS"] = "\n".join(rows)
    rep["EN_FRONTEND_LIST"] = '<ul class="dec">' + "".join(
        "<li><b>{}</b>: {}.</li>".format(i, TEXT["en_frontend"][i]) for i in tests_en) + "</ul>"

    # ------------------------------------------------------------------ Section 6.6 Telugu
    rows = []
    for i in tests_te:
        ra, rw = te_as[i], te_ws[i]
        c1, c2 = ((ra["asr_vs_model_input"]["cer"], rw["asr_vs_model_input"]["cer"]) if mode == "te_asr"
                  else (rw["asr_vs_model_input"]["cer"], rw["asr_script_mapped"]["cer"]))
        rows.append('<tr><td>{}</td><td class="n">{:.2f} s</td><td class="te">{}</td>'
                    '<td class="n">{}</td><td class="n">{}</td></tr>'.format(
                        i, rw["audio_s"], html.escape(ra.get("asr_transcript", "n/a")),
                        pct(c1), pct(c2)))
    rep["TE_SENTENCE_ROWS"] = "\n".join(rows)
    rep["TE_FRONTEND_LIST"] = '<ul class="dec">' + "".join(
        '<li><b>{}</b>: <span class="te">{}</span>.</li>'.format(i, TEXT["te_frontend"][i]) for i in tests_te) + "</ul>"

    # ------------------------------------------------------------------ Section 7 figures
    rep["TE_EN_SENTENCE_FIGS"] = ("<h3>7.1 English</h3>\n" + sentence_figs("English", tests_en, tests)
                                  + '\n<h3 class="pb">7.2 Telugu</h3>\n' + sentence_figs("Telugu", tests_te, tests))

    # ------------------------------------------------------------------ Section 8.3 / Section 8.4 tables
    def table(ids, lang, en_rows=None):
        cap = "{} {} sentences: Whisper transcripts and character error rate".format(
            "English" if lang == "en" else "Telugu", "stress" if ids[0][0] == "S" else "extended")
        cls = ' class="te"' if lang == "te" else ""
        if lang == "en":
            out = ['<table class="sx" data-cap="' + cap + '"><colgroup><col style="width:7%"><col style="width:14%"><col style="width:34%"><col style="width:35%"><col style="width:10%"></colgroup><tr><th>ID</th><th>Case</th><th>Text</th><th>Whisper large-v3 transcript</th>'
                   '<th class="n">CER norm.</th></tr>']
        elif mode == "te_asr":
            out = ['<table class="sx" data-cap="' + cap + '"><colgroup><col style="width:7%"><col style="width:14%"><col style="width:34%"><col style="width:35%"><col style="width:10%"></colgroup><tr><th>ID</th><th>Case</th><th>Text</th><th>Telugu-ASR transcript</th>'
                   '<th class="n">CER</th></tr>']
        else:
            out = ['<table class="sx" data-cap="' + cap + '"><colgroup><col style="width:7%"><col style="width:14%"><col style="width:34%"><col style="width:35%"><col style="width:10%"></colgroup><tr><th>ID</th><th>Case</th><th>Text</th><th>Whisper large-v3 transcript</th>'
                   '<th class="n">CER, script-mapped</th></tr>']
        for i in ids:
            meta = stress.get(i) or ext.get(i)
            r = en_s[i] if lang == "en" else te_as[i]
            m = r.get("asr_vs_original_whisper_norm") if lang == "en" else r.get("asr_vs_model_input")
            text = html.escape(texts[i]) + ('<br><span class="small">{}</span>'.format(html.escape(meta["gloss"]))
                                            if meta.get("gloss") else "")
            out.append('<tr><td>{}</td><td>{}</td><td{}>{}</td><td{}>{}</td><td class="n">{}</td></tr>'.format(
                i, html.escape(meta.get("category", "")), cls, text, cls, html.escape(r.get("asr_transcript") or "n/a"),
                pct(m["cer"]) if m else "n/a"))
        out.append("</table>")
        return "\n".join(out)

    se = ["SE0%d" % k for k in range(1, 9)]
    st = ["ST0%d" % k for k in range(1, 9)]
    xe = ["XE%02d" % k for k in range(1, 11)]
    xt = ["XT%02d" % k for k in range(1, 11)]
    rep["STRESS_TABLES"] = (
        "<h4>English (final configuration; CER after Whisper's English normaliser, against the original text): corpus {}</h4>".format(
            pct(corpus([en_s[i] for i in se], "asr_vs_original_whisper_norm")))
        + table(se, "en") + TEXT["stress_en_note"]
        + "<h4>Telugu (CER against the normalised text): corpus {}</h4>".format(
            pct(corpus([te_as[i] for i in st], "asr_vs_model_input")))
        + table(st, "te") + "STRESS_TE_NOTE")
    rep["EXTENDED_TABLES"] = (
        "<h4>English: corpus CER {}</h4>".format(pct(corpus([en_s[i] for i in xe], "asr_vs_original_whisper_norm")))
        + table(xe, "en") + TEXT["ext_en_note"]
        + "<h4>Telugu: corpus CER {}</h4>".format(pct(corpus([te_as[i] for i in xt], "asr_vs_model_input")))
        + table(xt, "te") + "EXT_TE_NOTE")

    for k in ("CLARITY_DECISION", "EN_OBSERVATIONS", "EN_FAILURES", "EN_INFERENCE_CMD"):
        rep[k] = TEXT[k]
    for k in ("TE_OBSERVATIONS", "TE_READING", "TE_FAILURES"):
        rep[k] = TEXT[k][mode]
    rep["STRESS_TABLES"] = rep["STRESS_TABLES"].replace("STRESS_TE_NOTE", TEXT["STRESS_TE_NOTE"][mode])
    rep["EXTENDED_TABLES"] = rep["EXTENDED_TABLES"].replace("EXT_TE_NOTE", TEXT["EXT_TE_NOTE"][mode])
    missing = [k for k, v in rep.items() if v is None or "STRESS_TE_NOTE" in v or "EXT_TE_NOTE" in v]
    if missing:
        raise SystemExit("report text missing for mode {}: {}".format(mode, missing))

    for k in sorted(rep, key=len, reverse=True):
        if k not in s:
            raise SystemExit("placeholder {} missing from report.html".format(k))
        s = s.replace(k, rep[k])
    left = sorted(set(re.findall(r"\b(?:EN|TE|TE2|CLARITY|STRESS|EXTENDED|EXT)_[A-Z0-9_]+\b", s)))
    if left:
        raise SystemExit("unfilled placeholders: " + ", ".join(left))
    s, lookup = number_and_index(s)
    render(s, browser)
    pages = find_pages(lookup)
    for key, pg in pages.items():  # second pass with the page numbers found in the first
        s = s.replace("{{PG:%s}}" % key, str(pg))
    s = re.sub(r"\{\{PG:[^}]+\}\}", "", s)
    render(s, browser)
    print("wrote", PDF, os.path.getsize(PDF), "bytes; page numbers", "filled" if pages else "not filled (pypdfium2 missing)")


def render(s, browser):
    with open(BUILT, "w", encoding="utf-8") as f:
        f.write(s)
    url = "file:///" + BUILT.replace("\\", "/")
    subprocess.run([browser, "--headless=new", "--disable-gpu", "--no-pdf-header-footer",
                    "--print-to-pdf=" + PDF, url], check=True, timeout=300)


def _plain(x):
    return " ".join(html.unescape(re.sub(r"<[^>]+>", "", x)).split())


def number_and_index(s):
    """Numbers the tables and figures marked with data-cap, gives every numbered h2/h3 an id, and writes the
    contents page and the lists of figures and tables. Returns the page and the (key, text to find) pairs whose
    page numbers find_pages looks up in the first rendering."""
    entries = {"toc": [], "fig": [], "tab": []}
    n = {"tab": 0, "fig": 0, "h": 0}

    def tab(m):
        n["tab"] += 1
        key = "tab%d" % n["tab"]
        entries["tab"].append((key, "Table %d. %s" % (n["tab"], m.group(2)), "Table %d." % n["tab"]))
        return '<table{}{} id="{}"><caption>Table {}. {}</caption>'.format(m.group(1), m.group(3), key, n["tab"], m.group(2))
    s = re.sub(r'<table([^>]*?) data-cap="([^"]*)"([^>]*)>', tab, s)

    def fig(m):
        n["fig"] += 1
        key = "fig%d" % n["fig"]
        entries["fig"].append((key, "Figure %d. %s" % (n["fig"], m.group(2)), "Figure %d." % n["fig"]))
        body = m.group(3).replace("<figcaption>", "<figcaption><b>Figure {}.</b> {}. ".format(n["fig"], m.group(2)), 1)
        return '<figure{} id="{}">{}</figure>'.format(m.group(1), key, body)
    s = re.sub(r'<figure([^>]*?) data-cap="([^"]*)">(.*?)</figure>', fig, s, flags=re.S)

    def head(m):
        attrs = m.group(2)
        if "nonum" in attrs:
            return m.group(0)
        n["h"] += 1
        key = "h%d" % n["h"]
        title = _plain(m.group(3))
        entries["toc"].append((key, m.group(1), title, title[:45]))
        return '<{0} id="{1}"{2}>{3}</{0}>'.format(m.group(1), key, re.sub(r'\s*id="[^"]*"', "", attrs), m.group(3))
    start = s.index("<!--FRONTMATTER-->")
    s = s[:start] + re.sub(r"<(h2|h3)([^>]*)>(.*?)</\1>", head, s[start:], flags=re.S)

    def line(key, label, cls):
        return ('<div class="tl {}"><a href="#{}">{}</a><span class="dots"></span><span class="pg">{{{{PG:{}}}}}</span></div>'
                .format(cls, key, label, key))
    out = ['<section class="front"><h2 class="nonum">Contents</h2>']
    out += [line(k, t, "l2" if lvl == "h2" else "l3") for k, lvl, t, _ in entries["toc"]]
    out.append('</section>\n<section class="front"><h2 class="nonum">List of Figures</h2>')
    out += [line(k, t, "l3") for k, t, _ in entries["fig"]]
    out.append('<h2 class="nonum lists">List of Tables</h2>')
    out += [line(k, t, "l3") for k, t, _ in entries["tab"]]
    out.append("</section>")
    s = s.replace("<!--FRONTMATTER-->", "\n".join(out))
    lookup = [(k, p) for k, _, _, p in entries["toc"]] + [(k, p) for k, _, p in entries["fig"] + entries["tab"]]
    return s, lookup


def find_pages(lookup):
    """Page of each heading/caption in the first rendering (pages after the lists are searched)."""
    try:
        import pypdfium2
    except ImportError:
        return {}
    doc = pypdfium2.PdfDocument(PDF)
    texts = [" ".join(doc[i].get_textpage().get_text_range().split()) for i in range(len(doc))]
    first = max(i for i, t in enumerate(texts) if "List of Tables" in t) + 1
    found = {}
    for key, probe in lookup:
        probe = " ".join(probe.split())
        for i in range(first, len(texts)):
            if probe in texts[i]:
                found[key] = i + 1
                break
    missing = [k for k, _ in lookup if k not in found]
    if missing:
        print("page not found for:", missing)
    return found


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--browser", default=r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe")
    main(ap.parse_args().browser)
