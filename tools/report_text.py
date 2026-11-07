"""Hand-written parts of the report (tools/build_report.py inserts them).

Every number here comes from a file in the repository:
  per-word durations and pitch: tools/word_stats.py --source model|final  (Results/<Lang>/variance_raw/<ID>.json,
  frontend_trace.json); rule actions: frontend_trace.json["prosody_rules"]; transcripts and CER:
  Evaluation/English/metrics_sentences.json, Evaluation/Telugu/asr_telugu/metrics_sentences.json,
  Evaluation/Telugu/metrics.json; configuration comparison: Evaluation/English/variants_summary.json.
Word duration = sum of its phones; word pitch = highest phone F0 of the word.
"""

ANALYSIS = {
    # ------------------------------------------------------------------ English: model prediction first, then the options
    "E1": [
        "<b>Duration (model).</b> <i>antidisestablishmentarianism</i> takes 2,020 ms (26 % of the model's 7.74 s) with uniformly short phones; "
        "<i>parliamentary</i> 801 ms, <i>proceedings</i> 882 ms. The final vowel is only 1.1× the mean vowel: little phrase-final lengthening. "
        "At the two micro-pause positions the model predicts 383 and 464 ms, as long as phrase-boundary pauses; the option "
        "sets them to 70 ms.",
        "<b>Pitch (model).</b> Several similar peaks (<i>considerable</i> 261, <i>antidisestablishmentarianism</i> 256, <i>movement</i> 248 Hz), "
        "lowest on <i>during</i> (204 Hz), first third 225 → last third 195 Hz: several peaks of similar height and an overall fall.",
        "<b>Energy (model).</b> Vowels median 63, nasals 43, other consonants 15 a.u., pauses ≈ 3; the final vowel of <i>proceedings</i> is 41.",
        "<b>Options.</b> Long words × 1.25 and everything × 1.10: the long word 2,020 → 2,798 ms, <i>parliamentary</i> 801 → 1,103 ms; the stress "
        "rule raised the stressed vowel of seven words (pitch up to × 1.12, energy up to × 1.5). Whisper recognises the long word exactly.",
    ],
    "E2a": [
        "<b>Duration (model).</b> <i>conduct</i> (noun, <code>K AA1 N D AH0 K T</code>): syllables 313 and 325 ms (ratio 0.96), the longest word "
        "of the sentence (639 ms). <i>audience</i> is long (662 ms) but its final vowel is short (0.9× the mean) because the word ends in /s/.",
        "<b>Pitch (model).</b> The stressed first vowel is <i>lower</i> than the unstressed second (211 vs 224 Hz): the model does not mark the "
        "noun's stress, in line with the data (Section 6, patterns). Main accents on <i>orchestra's</i> and <i>ceremony</i> (241 Hz); first third "
        "220 → last third 201 Hz.",
        "<b>Energy (model).</b> Vowels median 74, nasals 46, other consonants 22 a.u.; highest on the /iː/ of the article before "
        "<i>orchestra's</i> (135).",
        "<b>Options.</b> The stress rule makes the first vowel of <i>conduct</i> the higher (233 vs 224 Hz) and the louder (× 1.5), with syllables "
        "360 / 348 ms; the terminal fall lowers the last two phones to 182 Hz; the micro-pause before <i>the ceremony</i> is 70 ms (model: 255 ms).",
    ],
    "E2b": [
        "<b>Duration (model).</b> <i>conduct</i> (verb, <code>K AA0 N D AH1 K T</code>; the input differs from the noun only in the stress digits): syllables 279 / 279 ms (ratio 1.00, against 0.96 for the "
        "noun) and 557 ms in all (noun: 639 ms). In the model's prediction the noun/verb difference is mainly the word's length, not which syllable "
        "is prominent. Final lengthening 1.2× on <i>method</i>.",
        "<b>Pitch (model).</b> Vowels of <i>conduct</i> 215 / 218 Hz (almost level); peaks on <i>will</i> (251 Hz) and <i>researchers</i> "
        "(237 Hz); lowest on <i>an</i> (189 Hz); first third 224 → last third 198 Hz.",
        "<b>Energy (model).</b> Vowels median 72, nasals 44, other consonants 12 a.u.; the strongest phone is the /ɚ/ of <i>researchers</i> (121).",
        "<b>Options.</b> The stress rule puts the higher pitch on the stressed second vowel (224 vs 215 Hz). With E2a (233 vs 224 Hz on the "
        "first vowel) the final audio therefore carries the homograph contrast in pitch, which the model alone hardly does. Whisper: 10.9 % CER "
        "with the post-filter only (configuration R0, earlier frontend: \"an experimental evaluation of\"), exact with the final options.",
    ],
    "E3": [
        "<b>Duration (model).</b> The split word is pronounced as <i>rhythm</i> (<code>R IH1 DH AH0 M</code>, 383 ms), without a pause inside it: "
        "the double space has no effect after joining. <i>unexpectedly</i> is very long (1,149 ms) with a final vowel of 221 ms (2.5× the mean), "
        "the strongest final lengthening of the five sentences.",
        "<b>Pitch (model).</b> The model ends this statement on its highest pitch: the last vowel of <i>unexpectedly</i> is 269 Hz, a "
        "question-like rise, while <i>rhythm</i> and <i>of</i> carry 244–248 Hz.",
        "<b>Energy (model).</b> Vowels median 68, nasals 52, other consonants 16 a.u.; the /r/ of <i>orchestra</i> is the energy maximum (108), as an "
        "approximant with vowel-like energy.",
        "<b>Options.</b> The terminal-fall rule lowers the last two vowels from 201 and 269 Hz to 185 Hz, turning the rise into a statement fall; "
        "<i>unexpectedly</i> 1,149 → 1,544 ms (long word); the stress rule adds energy to the first vowel of <i>orchestra</i> (× 1.5).",
    ],
    "E4": [
        "<b>Duration (model).</b> Numbers read as words with growing durations (<i>ten</i> 232, <i>forty</i> 325, <i>five</i> 395 ms; "
        "<i>twelve</i> 441, <i>November</i> 464, <i>six</i> 441 ms): each number group ends on its longest word; <i>twenty twenty</i> 337 + 325 ms; "
        "<i>AM</i> 290 ms. The comma gives a 406 ms pause. After it the function words are very short (<i>should</i> 232 ms with a 70 ms "
        "vowel, <i>on</i> 151 ms); final lengthening 1.8× on <i>time</i>.",
        "<b>Pitch (model).</b> Highest on <i>meeting</i> (268 Hz), lowest on <i>time</i> (188 Hz); no pitch reset after the comma "
        "(<i>six</i> 216 → <i>so</i> 219 Hz), unlike Telugu T4; first third 237 → last third 205 Hz.",
        "<b>Energy (model).</b> Vowels median 64, nasals 46, other consonants 13 a.u.; /s/ of <i>six</i> and <i>so</i> ≈ 5 (fricatives), "
        "the comma pause 1.4.",
        "<b>Options.</b> <i>AM</i> as letter names × 1.6 (× 1.10): 290 → 499 ms; comma pause 441 ms; <i>twenty</i> in its citation form with /t/; "
        "the stress rule lengthens the stressed vowels of <i>November</i> (× 1.4), the first <i>twenty</i> (× 1.31) and <i>andaroo</i> (× 1.24). The last clause remains the weak point "
        "for the ASR (Section 5.4).",
    ],
    # ------------------------------------------------------------------ Telugu (as submitted, no options)
    "T1": [
        "<span class=\"te\">న్యూరోఫార్మకోలాజికల్</span> lasts 1,695 ms (32 % of 5.27 s) and carries the sentence's highest pitch (313 Hz).",
        "Pitch drops to 230 Hz on <span class=\"te\">పరిశోధనలో</span> and stays in a narrow 237–258 Hz band for the rest: the F0 maximum is early and the sentence-final verb is low.",
        "The final verb <span class=\"te\">విశ్లేషించారు</span> is long (964 ms; cluster <code>sh l</code>, nasal <code>ny</code> before <code>c</code>) with 1.3× final lengthening; no internal pauses (no internal punctuation).",
    ],
    "T2a": [
        "Peak pitch on the first word <span class=\"te\">తాడుపై</span> (326 Hz), then a steady fall to <span class=\"te\">ఉంది</span> (207 Hz).",
        "<span class=\"te\">పట్టు</span> (<code>p a tt tt u</code>, geminate retroflex) lasts 337 ms with 242 Hz, inside the falling line, not accented.",
        "Final vowel 151 ms, 1.8× the mean: strong phrase-final lengthening.",
    ],
    "T2b": [
        "<span class=\"te\">పట్టు</span> has the same phones and the same duration (337 ms) as in T2a, as it should (the two words are written and pronounced alike), but now carries the sentence's F0 maximum (295 Hz vs 242 Hz in T2a), directly before <span class=\"te\">చీర</span> and after a low start on <span class=\"te\">ఆమె</span> (220 Hz).",
        "The rest declines from 254 Hz (<span class=\"te\">చీరను</span>) to 202 Hz (<span class=\"te\">వచ్చింది</span>), with 1.7× final lengthening.",
        "The input phones of the word are identical in T2a and T2b; the model gives it the same duration and a different F0 depending on its position in the sentence (there is no recording of these sentences to compare with).",
    ],
    "T3": [
        "Peak on <span class=\"te\">పులి</span> (321 Hz, the topic).",
        "The broken word <span class=\"te ws\">అడ  విలో</span> gives exactly the phone sequence of <span class=\"te\">అడవిలో</span> (<code>a dd a v i l oo</code>; checked), because the Telugu input has no word-boundary symbol: the model receives the intended word. It is spoken without a pause (<span class=\"te\">అడ</span> 186 ms, <span class=\"te\">విలో</span> 279 ms, contiguous), with pitch rising from 217 to 274 Hz across the word.",
        "<span class=\"te\">నిశ్శబ్దంగా</span> is long (778 ms; geminate <code>sh sh</code> and the cluster <code>b d</code>); <span class=\"te\">నడిచింది</span> ends the sentence at the pitch minimum (194 Hz) with 2.1× final lengthening.",
    ],
    "T4": [
        "Longest sentence (9.13 s, 120 phones). Numbers are expanded to Telugu words: <span class=\"te\">రెండు వేల ఇరవై ఆరు</span> (302/290/453/267 ms), <span class=\"te\">పన్నెండవ</span> (615 ms), <span class=\"te\">పది నలభై ఐదు</span> (244/418/267 ms), read without hesitation pauses.",
        "The comma gives a 546 ms pause followed by a clear <b>pitch reset</b>: from ≈ 208 Hz on <span class=\"te\">ప్రారంభమవుతుంది</span> (the longest word, 1,126 ms) up to 270 Hz on <span class=\"te\">సమయానికి</span>: the F0 of the second clause starts high again (a reset).",
        "Highest pitch on <span class=\"te\">సమావేశం</span> (272 Hz), lowest on the final <span class=\"te\">ఉంచండి</span> (198 Hz), with 1.5× final lengthening.",
    ],
}
# Energy (a.u., per-phone STFT-norm energy; medians by phone class), Telugu: the model's prediction as used.
_TE_ENERGY = {
    "T1": "Vowels median 79, nasals 65, other consonants 35; the final /u/ of <span class=\"te\">విశ్లేషించారు</span> falls to 15: "
          "the energy falls over the last syllables.",
    "T2a": "Vowels median 82, nasals 72, other consonants 19; the geminate <code>tt tt</code> of <span class=\"te\">పట్టు</span> is the "
           "energy minimum (8.6), the long closure of the stop; the final /i/ drops to 25.",
    "T2b": "Vowels median 84, nasals 88: the /m/ of <span class=\"te\">ఆమె</span> is the strongest phone (139), so the sentence starts "
           "loud; other consonants 19; the final /i/ drops to 26.",
    "T3": "Vowels median 86, nasals 60, other consonants 35; strongest on an /i/ (141); the final /i/ of "
          "<span class=\"te\">నడిచింది</span> drops to 26.",
    "T4": "Vowels median 75, nasals 74, other consonants 37; the comma pause is near silence (0.3); the final /i/ drops to 28. In all five "
          "Telugu sentences the last vowel has 15–28 against a vowel median of 75–86, a stronger end-of-sentence decay than in English "
          "(41–55 against 63–74).",
}
for _k, _v in _TE_ENERGY.items():
    ANALYSIS[_k].append("<b>Energy.</b> " + _v)

TEXT = {
    "en_frontend": {
        "E1": "\"The\" before a vowel → <code>DH IY0</code>; <i>antidisestablishmentarianism</i> is not in the lexicon → split into lexicon words "
              "anti + disestablishment + arianism (<code>AE1 N T AY0 D IH0 S IH0 S T AE1 B L IH0 SH M AH0 N T AE1 R IY0 AH0 N IH2 Z AH0 M</code>); "
              "<i>parliamentary</i> in its citation form (restored /t/); micro-pauses before <i>considerable</i> and <i>the parliamentary</i>",
        "E2a": "<i>conduct</i> tagged noun (NN) → <code>K AA1 N D AH0 K T</code>; micro-pause before <i>the ceremony</i>",
        "E2b": "<i>conduct</i> tagged verb (VB) → <code>K AA0 N D AH1 K T</code>; micro-pauses before <i>an experiment</i> and <i>to evaluate</i>",
        "E3": "\"<span class=\"ws\">rhyth  m</span>\" joined for pronunciation → <i>rhythm</i> <code>R IH1 DH AH0 M</code> (text unchanged); micro-pause before <i>unexpectedly</i>",
        "E4": "\"ten forty five A M on twelve November twenty twenty six\"; AM as letter names (durations × 1.6); <i>twenty</i> in its citation form "
              "(restored /t/); <i>andaroo</i> (Telugu అందరూ, \"everyone\") → <code>AE0 N D AA1 R UW0</code>",
    },
    "te_frontend": {
        "T1": "all words by rule; <code>ny</code> before <code>c</code> in విశ్లేషించారు",
        "T2a": "పట్టు → <code>p a tt tt u</code>",
        "T2b": "పట్టు → <code>p a tt tt u</code> (same as T2a)",
        "T3": "the split <span class=\"ws\">అడ  విలో</span> gives the phones of అడవిలో; no pause between the parts",
        "T4": "2026 → రెండు వేల ఇరవై ఆరు; 12వ → పన్నెండవ; 10:45 → పది నలభై ఐదు",
    },
    # ------------------------------------------------------------------ Telugu, per ASR mode
    # "whisper": Whisper large-v3 only (Evaluation/Telugu/metrics.json, metrics_extended_whisper.json);
    # "te_asr": with the Telugu ASR of Kaggle job fs2-20-asr-te-cpu (Evaluation/Telugu/asr_telugu/).
    "TE_ASR_BOX": {
        "whisper": "Whisper large-v3 [8] for both languages; for Telugu, transcripts that Whisper wrote in Kannada script are also "
                   "scored after mapping them back to Telugu, Section 5.5",
        "te_asr": "Whisper large-v3 for English; for Telugu also a Telugu Whisper model not trained on IndicTTS, Section 7",
    },
    "TE_ASR_NOTE": {
        "whisper": "Whisper large-v3 is weak on Telugu: its CER on copy-synthesis of real Telugu validation recordings is 33.1 %, "
                   "against 11.6 % for English, so Telugu CERs bound pronunciation errors from above rather than measure them. "
                   "A Telugu ASR model (<code>vasista22/whisper-telugu-large-v2</code> [17], not trained on IndicTTS; pinned in "
                   "<code>evaluate_val.ASR_REVISIONS</code>) was prepared for this evaluation (Kaggle <code>fs2-20-asr-te-cpu</code>), "
                   "but it runs on CPU (the weekly GPU quota was used up) and had not finished when this report was built.",
        "te_asr": "Telugu ASR: <code>vasista22/whisper-telugu-large-v2</code> (Whisper large-v2 fine-tuned on Telugu; its model card "
                  "lists CSTD IIIT-H, ULCA, Shrutilipi, Microsoft Speech Corpus, FLEURS and Babel as training data, not IndicTTS, so it "
                  "has not heard the Telugu training voices; FLEURS-te test WER 9.65 %), revision <code>ba19a24</code>, on evenly "
                  "spaced validation clips (Kaggle <code>fs2-20-asr-te-cpu</code>, on CPU because the weekly GPU quota was used up). "
                  "Whisper large-v3 is weak on Telugu even on copy-synthesis of real recordings, which is why the second ASR was added.",
    },
    "TE_OBSERVATIONS": {
        "whisper": (
            "<p><b>Observations.</b> All five sentences pass the frontend with every number expanded to Telugu words. Whisper "
            "large-v3's transcripts show what is recognised, with its weak Telugu in mind (copy-synthesis ceiling 33.1 %, Section 7): "
            "<b>T1</b>: the English loanword is recognised and written in Latin script (<i>Neuro-Pharmacological</i>), which is most "
            "of the CER; the rest is శ → స in <span class=\"te\">పరిశోధనలో</span> and <span class=\"te\">విశ్లేషించారు</span> and "
            "ఔ → ఉ in <span class=\"te\">ఔషధాల</span>. <b>T2a/T2b</b>: <span class=\"te\">పట్టు</span> is transcribed correctly in both "
            "senses (written and pronounced alike; only the prosody differs, Section 6.2); the errors are word-boundary merges "
            "(<span class=\"te\">తాడు పయతని</span> for <span class=\"te\">తాడుపై అతని</span>, <span class=\"te\">ఆమేపట్టు</span>) and "
            "vowel length (<span class=\"te\">ఆమె → ఆమే</span>, <span class=\"te\">చీర → చిర</span>). <b>T3</b>: the split "
            "<span class=\"te ws\">అడ  విలో</span> is transcribed by Whisper as the single word <span class=\"te\">అడవిలో</span> "
            "(no pause between the parts, Section 6.2); the only error is again శ → స. <b>T4</b>: Whisper writes the spoken numbers back as digits (2026, "
            "10.45), which the CER counts as errors, drops the opening <span class=\"te\">ఈ సమావేశం</span>, merges "
            "<span class=\"te\">నవంబర్ పన్నెండవ తేదీ</span> across word boundaries (the syllables are present) and hears "
            "<span class=\"te\">ప్రారంభ</span> without aspiration; the final <span class=\"te\">ఉంచండి</span> is exact.</p>"
            "<p>Across the 22 distinct Telugu sentences scored (Sections 5.5, 7.3, 7.4; XT01 repeats T1) the same confusions recur: శ appears "
            "in the transcripts for only 2 of its 16 occurrences (mostly written స), ణ for 1 of 6 (written న), and 16 of 26 aspirated "
            "consonants are kept (<span class=\"te\">కథ → కద</span>, <span class=\"te\">పాఠకులు → పాదకులు</span>); some losses are "
            "words the ASR dropped entirely. "
            "The frontend keeps all these phones distinct, so the merger is in the acoustic model, in Whisper, or both; with this "
            "ASR the two cannot be separated <span class=\"tag est\">interpretation</span>.</p>"),
        "te_asr": None,
    },
    "TE_READING": {
        "whisper": "(4) For Telugu, Whisper large-v3 loses nearly as much on copy-synthesis of real recordings (33.1 %) as on our "
                   "synthesis (35.8 %): as far as this ASR can tell, the acoustic model costs 2.7 points of CER.",
        "te_asr": None,
    },
    "TE_FAILURES": {
        "whisper": "శ heard as స, ణ as న, lost aspiration (model or ASR, not separable here); word-boundary merges; T4's spoken "
                   "numbers written back as digits by the ASR; no question intonation (XT09, Section 8.2)",
        "te_asr": None,
    },
    "STRESS_TE_NOTE": {
        "whisper": "<p class=\"small\">ST06: the Latin-script <i>iPhone</i> is transcribed <span class=\"te\">ఆఈ ఫోన్</span>, i.e. "
                   "spoken as intended through <code>text/telugu/latin.py</code>. Whisper writes digits where it hears none "
                   "(\"3.\" for <span class=\"te\">స్త్రీ</span> in ST04, \"4\" for <span class=\"te\">నా</span> in ST05) and drops "
                   "question and exclamation marks (ST01, ST08), which the CER ignores.</p>",
        "te_asr": None,
    },
    "EXT_TE_NOTE": {
        "whisper": "<p class=\"small\">Whisper wrote XT05, XT09 and XT10 in Kannada script (76–107 Kannada characters each; raw CER "
                   "100–113 %); the table scores them after mapping the Kannada block back onto the Telugu block (same layout, "
                   "offset 0x80; <code>tools/build_report.py</code>). XT09's question particle <span class=\"te\">కాదా</span> is "
                   "recognised; its intonation falls (Section 8.2). XT04 and XT01 repeat T4's and T1's ASR effects (digits, Latin script). "
                   "Whisper transcripts: <code>Evaluation/Telugu/metrics_extended_whisper.json</code> (Kaggle "
                   "<code>fs2-14-extended</code>).</p>",
        "te_asr": None,
    },
}

_FINAL_FLAGS = ("--gv_ratio Results/English/gv_ratio.json --gv_strength 1.0 --prosody_fall --prosody_stress 1.0 "
                "--prosody_phrasing --duration_control 1.10 --long_word_scale 1.25 --heavy_word_pause "
                "--denoise_profile Results/English/noise_profile.json --edge_fade")
BS = chr(92)  # line continuation in the shell commands below

TEXT.update({
    "CLARITY_DECISION": (
        "<p><b>Which combination.</b> Every configuration was rendered on Kaggle and scored with Whisper large-v3 on the five test "
        "sentences and 18 further sentences (Sections 7.3–7.4: 8 stress and 10 extended; XE01 is E1 verbatim, so E1 counts twice in \"all 23\"); "
        "CER after Whisper's English normaliser; the five test sentences total 335 reference characters, so one character is 0.3 points. "
        "The table is <code>Evaluation/English/variants_summary.json</code>:</p>"
        "<table><tr><th>English configuration (same checkpoint and voice)</th><th class=\"n\">CER, 5 test sentences</th>"
        "<th class=\"n\">CER, all 23</th><th class=\"n\">E4</th></tr>"
        "<tr><td>model + GV post-filter †</td><td class=\"n\">2.7 %</td><td class=\"n\">2.9 %</td><td class=\"n\">2.9 %</td></tr>"
        "<tr><td>+ terminal fall †</td><td class=\"n\">2.7 %</td><td class=\"n\">3.2 %</td><td class=\"n\">2.9 %</td></tr>"
        "<tr><td>+ fall + lexical stress</td><td class=\"n\">6.9 %</td><td class=\"n\">4.1 %</td><td class=\"n\">17.1 %</td></tr>"
        "<tr><td>+ fall + stress, denoise, edge fade</td><td class=\"n\">7.8 %</td><td class=\"n\">4.0 %</td><td class=\"n\">15.7 %</td></tr>"
        "<tr><td><b>final: fall + stress + phrasing, duration × 1.10, long words × 1.25 with micro-pauses, denoise, edge fade</b></td>"
        "<td class=\"n\"><b>3.6 %</b></td><td class=\"n\"><b>2.3 %</b></td><td class=\"n\">17.1 %</td></tr>"
        "<tr><td>final without the lexical-stress rule (Kaggle <code>fs2-21-final-en-nostress</code>)</td><td class=\"n\">3.9 %</td>"
        "<td class=\"n\">2.3 %</td><td class=\"n\">15.7 %</td></tr></table>"
        "<p class=\"small\">† Earlier frontend (before the citation forms, the \"the\" rule, the compound split and the acronym rules), so "
        "these two rows are not strictly comparable with the others. Kaggle jobs: <code>fs2-12-prosody-en</code>, <code>fs2-14-extended</code>, "
        "<code>fs2-16-clarity</code>, <code>fs2-17-final-en</code>, <code>fs2-21-final-en-nostress</code>.</p>"
        "<p>The final configuration has the lowest error over all 23 sentences (differences of under one point are a few characters and not "
        "significant), and the author preferred it on listening over the earlier configurations. "
        "Removing the stress rule changes little (2.28 → 2.34 %); the rule is kept because it puts the homograph contrast of E2a/E2b into "
        "pitch (Section 6.1). One sentence is worse than with the plain model: the last clause of E4 (\"so andaroo should arrive on time\") is "
        "garbled for the ASR in every configuration with the current frontend (15.7–17.1 %), with or without the stress rule, whereas "
        "the model with only the post-filter and the earlier frontend gave \"So, Antaro should arrive on time\" (2.9 %). The cause was not "
        "isolated; the stretch is a made-up word followed by function words that the model predicts very short (Section 6.1, E4).</p>"),
    "EN_OBSERVATIONS": (
        "<p><b>Observations.</b> Four of the five sentences are transcribed exactly (E4's raw CER is mostly digit formatting: \"10.45\", "
        "\"12\", \"2026\"). <b>E1</b>: the long rare word is recognised as the intended word; it needed the compound split and the slower "
        "long-word timing. <b>E2a/E2b</b>: the homograph pair is transcribed correctly, with the stress contrast in pitch (Section 6.1). "
        "<b>E3</b>: exact, including the joined <i>rhythm</i>. <b>E4</b>: the numbers, the date and <i>AM</i> are recognised; the last "
        "clause is not (\"Suhant Arashudra I. One time.\"): <i>andaroo</i> is a Telugu word (అందరూ, \"everyone\") that the ASR cannot "
        "know, and the function words after it are short (Section 5.3). Across the five sentences the clear-speech options make the endings "
        "complete and the pauses quieter; on listening (the author's impression; no listening test was run) short unstressed function words and a slightly "
        "breathy quality remain (Section 5.2).</p>"
        "<p><b>Found by inspecting the outputs.</b> The first full English run (same checkpoint) read E3 as <code>R AY1 DH</code> + the "
        "letter name <code>EH1 M</code> (\"rythe em\") and ended <i>andaroo</i> in <code>OW</code>. Both were caught in "
        "<code>frontend_trace.json</code> and fixed by general rules, covered by tests (Section 1.2); the outputs were regenerated with the "
        "unchanged checkpoint.</p>"),
    "EN_FAILURES": ("E4's last clause (a made-up word followed by very short function words) is garbled for the ASR; short "
                    "unstressed function words; lexical stress, question rises and statement falls come from rules, not from the "
                    "model (Sections 5.3, 6); the muffled first test voice (Section 5.1)"),
    "EN_INFERENCE_CMD": (
        "python tools/noise_profile.py --config English/config.yaml --data_root D " + BS + "\n"
        "       --checkpoint English/checkpoints/en_model.pth.tar --speaker svarah_8a8e2655 " + BS + "\n"
        "       --gv_ratio Results/English/gv_ratio.json --gv_strength 1.0 --out Results/English/noise_profile.json\n"
        "python English/inference.py  --data_root D --checkpoint English/checkpoints/en_model.pth.tar " + BS + "\n"
        "       --speaker svarah_8a8e2655 --gv_ratio Results/English/gv_ratio.json --gv_strength 1.0 " + BS + "\n"
        "       --prosody_fall --prosody_stress 1.0 --prosody_phrasing --duration_control 1.10 " + BS + "\n"
        "       --long_word_scale 1.25 --heavy_word_pause --denoise_profile Results/English/noise_profile.json " + BS + "\n"
        "       --edge_fade --out_dir Results/English\n"
        "# gv_ratio.json: fastspeech2/postfilter_eval.py (Kaggle fs2-08-postfilter). Without the options: the model alone."),
    "stress_en_note": ("<p class=\"small\">SE05: \"elliptic curve\" heard as \"elliptical\"; SE06: the spelled acronym IIIT heard as "
                       "\"IIT\" and <i>Rao</i> as \"Lau\".</p>"),
    "ext_en_note": ("<p class=\"small\">XE06's remaining CER is formatting: the ASR writes \"37.5°C\", which its normaliser does not "
                    "map to \"degrees Celsius\"; the content is right. XE02: \"The orchestra's conduct during\" heard as \"Bay "
                    "Orchestra's conducting\". Audio: <code>Evaluation/Extended/English/</code>.</p>"),
})

