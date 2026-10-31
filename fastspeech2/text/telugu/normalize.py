"""Telugu text normalisation for the TTS frontend.

Only produces the model-input text; the original text is never modified in place.
Rules (each one is listed in the report):
  1. Unicode NFC.
  2. ZERO WIDTH JOINER / NON-JOINER are removed (they only affect glyph shaping).
  3. Telugu digits (U+0C66..U+0C6F) are mapped to ASCII digits.
  4. "H:MM" times are read as "<hours> <minutes>" (10:45 -> పది నలభై ఐదు).
  5. A number followed by the ordinal suffix వ (e.g. 12వ) is read as an ordinal (పన్నెండవ).
  6. Other numbers 0..9999 are read as Telugu cardinals; larger numbers raise an error.
  7. Runs of whitespace collapse to one space (the word boundary itself is preserved).

The number words below were written from general knowledge of Telugu and must be reviewed
by a native speaker before the results are reported (plan.txt, risk R4).
"""
import re
import unicodedata

UNITS = [
    "సున్నా", "ఒకటి", "రెండు", "మూడు", "నాలుగు", "ఐదు", "ఆరు", "ఏడు", "ఎనిమిది", "తొమ్మిది",
    "పది", "పదకొండు", "పన్నెండు", "పదమూడు", "పద్నాలుగు", "పదిహేను", "పదహారు", "పదిహేడు",
    "పద్దెనిమిది", "పందొమ్మిది",
]
TENS = {2: "ఇరవై", 3: "ముప్పై", 4: "నలభై", 5: "యాభై", 6: "అరవై", 7: "డెబ్బై", 8: "ఎనభై", 9: "తొంభై"}
HUNDRED, HUNDREDS_OF = "వంద", "వందల"  # 100 / "<n> hundred and ..."
HUNDREDS_EXACT = "వందలు"  # "<n> hundred" with nothing after it
THOUSAND, THOUSANDS_OF = "వెయ్యి", "వేల"

_TIME_RE = re.compile(r"(?<!\d)(\d{1,2}):(\d{2})(?!\d)")
_ORDINAL_RE = re.compile(r"(\d+)వ(?![ా-్])")  # 12వ but not 12వా / 12వి ...
_NUMBER_RE = re.compile(r"\d+")
_WS_RE = re.compile(r"\s+")


def _below_100(n):
    if n < 20:
        return UNITS[n]
    tens, unit = divmod(n, 10)
    return TENS[tens] if unit == 0 else f"{TENS[tens]} {UNITS[unit]}"


def _below_1000(n):
    if n < 100:
        return _below_100(n)
    hundreds, rest = divmod(n, 100)
    if hundreds == 1:
        head = HUNDRED  # "వంద", "వంద ఇరవై"
    else:
        head = f"{UNITS[hundreds]} {HUNDREDS_EXACT if rest == 0 else HUNDREDS_OF}"
    return head if rest == 0 else f"{head} {_below_100(rest)}"


def cardinal(n):
    """Telugu cardinal number words for 0 <= n <= 9999."""
    if not 0 <= n <= 9999:
        raise ValueError(f"number out of supported range 0..9999: {n}")
    if n < 1000:
        return _below_1000(n)
    thousands, rest = divmod(n, 1000)
    head = THOUSAND if thousands == 1 else f"{UNITS[thousands]} {THOUSANDS_OF}"
    return head if rest == 0 else f"{head} {_below_1000(rest)}"


def ordinal(n):
    """Ordinal form used with the suffix వ: drop the final vowel sign of the cardinal, add వ."""
    words = cardinal(n).split(" ")
    last = words[-1]
    if last[-1] in ("ు", "ి"):  # final ు or ి (రెండు -> రెండవ, పది -> పదవ)
        last = last[:-1]
    words[-1] = last + "వ"
    return " ".join(words)


def normalize(text):
    """Return the model-input form of a Telugu sentence (see module docstring)."""
    text = unicodedata.normalize("NFC", text)
    text = text.replace("‌", "").replace("‍", "")
    text = text.translate({0x0C66 + i: str(i) for i in range(10)})
    text = _TIME_RE.sub(lambda m: f"{cardinal(int(m.group(1)))} {cardinal(int(m.group(2)))}", text)
    text = _ORDINAL_RE.sub(lambda m: ordinal(int(m.group(1))), text)
    text = _NUMBER_RE.sub(lambda m: cardinal(int(m.group(0))), text)
    return _WS_RE.sub(" ", text).strip()
