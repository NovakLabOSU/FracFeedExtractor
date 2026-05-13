"""Coarse relevance filtering for cleaned scientific-paper text.

Runs between ``text_cleaner.clean_text()`` and
``llm_text.extract_key_sections()`` to drop entire paragraphs that are very
unlikely to contain any of the target extraction metrics:

  - predator species names
  - study locations
  - collection dates
  - sample sizes (n=, stomachs examined)
  - empty / non-empty stomach counts
  - feeding fraction data

Paragraphs about pure taxonomy debates, phylogenetic analysis, habitat ecology
without location data, literature reviews of other studies, detailed prey-ID
methodology, morphometric measurements, or statistical test descriptions are
dropped.

**Conservative by design** — a paragraph is kept unless it matches at least one
negative pattern *and* scores zero on every positive pattern.  Borderline
paragraphs are always retained.

Exposes one primary function::

    filter_relevant_sections(text: str) -> str

The function preserves ``[PAGE N]`` markers and section headings so the
downstream ``extract_key_sections()`` can still do its section-priority ranking.
"""

import re
from typing import List, Tuple

# ---------------------------------------------------------------------------
# Positive (keep) patterns — if ANY matches, the paragraph is kept regardless
# of negative signals.  Mirrors the field patterns in llm_text.py but cast as
# a binary keep gate rather than a weighted scorer.
# ---------------------------------------------------------------------------

_POSITIVE_PATTERNS: List[re.Pattern] = [
    # sample_size — explicit counts of stomachs / specimens / individuals
    re.compile(
        r"(?i)(\bn\s*=\s*\d+"
        r"|total\s+of\s+\d+"
        r"|\d+\s+stomachs?"
        r"|\d+\s+specimens?"
        r"|\d+\s+individuals?"
        r"|\d+\s+birds?"
        r"|\d+\s+fish"
        r"|\d+\s+samples?"
        r"|sample\s+size\s+(of\s+)?\d+"
        r"|examined\s+\d+"
        r"|\d+\s+(were|was)\s+(examined|collected|analysed|analyzed|sampled))"
    ),
    # empty-stomach language
    re.compile(
        r"(?i)(empty\s+stomachs?"
        r"|stomachs?\s+(were\s+)?empty"
        r"|had\s+empty"
        r"|without\s+food"
        r"|without\s+(stomach\s+)?contents?"
        r"|zero\s+prey"
        r"|no\s+food\s+(items?|remains?)"
        r"|vacuous|vacant\s+stomachs?)"
    ),
    # non-empty / fraction feeding
    re.compile(
        r"(?i)(non.?empty"
        r"|contained\s+(food|prey|items?)"
        r"|with\s+food"
        r"|with\s+(stomach\s+)?contents?"
        r"|had\s+(food|prey)"
        r"|feeding\s+rate"
        r"|proportion\s+(feeding|with\s+food)"
        r"|percent\s+(feeding|with\s+food)"
        r"|\d+\s*%\s+of\s+(stomachs?|individuals?|birds?|fish|specimens?))"
    ),
    # percentage / fraction near gut/stomach context
    re.compile(r"(?i)(\d+\.?\d*\s*%|\d+\s+percent" r"|\d+\s+of\s+\d+\s+(were|had|contained)" r"|proportion\s+of\s+\d+)"),
    # study date — collection period
    re.compile(
        r"(?i)(collected\s+(in|during|between)"
        r"|sampled\s+(in|during|between)"
        r"|field\s+season"
        r"|study\s+period"
        r"|between\s+\d{4}\s+and\s+\d{4}"
        r"|\d{4}[\-\u2013]\d{4}"
        r"|sampling\s+period)"
    ),
    # study location
    re.compile(r"(?i)(study\s+(area|site|region)" r"|specimens?\s+(were\s+)?(obtained|collected|caught)\s+(from|at|in)" r"|sampling\s+(location|site|area))"),
    # gut / stomach / diet core vocabulary
    re.compile(
        r"(?i)(stomach\s+content"
        r"|diet\s+(composition|analysis|study)"
        r"|gut\s+content"
        r"|food\s+(items?|habits?|composition)"
        r"|prey\s+(items?|composition|species|frequency)"
        r"|trophic\s+(level|ecology|niche)"
        r"|feeding\s+(ecology|habits?|behaviour|behavior)"
        r"|gastrointestinal|foregut|hindgut|crop\s+content)"
    ),
    # predator species name — binomial in a sentence with diet/food/stomach
    re.compile(r"(?i)\b[A-Z][a-z]+\s+[a-z]{3,}\b.{0,80}" r"(stomach|diet|prey|food|feeding|gut|trophic|forag)"),
    # geographic coordinates or explicit lat/lon
    re.compile(r"(\d{1,3}[°º]\s*\d{0,2}['′]?\s*[NS]" r"|\d{1,3}[°º]\s*\d{0,2}['′]?\s*[EW]" r"|latitude|longitude" r"|\d+\.\d+\s*[°º]?\s*[NS],?\s*\d+\.\d+\s*[°º]?\s*[EW])"),
    # table-like numeric data (rows of numbers separated by whitespace/tabs)
    re.compile(r"(?m)^.*(\d+\s*[\t|]\s*){2,}\d+"),
]

# ---------------------------------------------------------------------------
# Negative (drop-candidate) patterns — a paragraph is dropped ONLY when it
# matches at least one negative pattern AND matches ZERO positive patterns.
# ---------------------------------------------------------------------------

_NEGATIVE_PATTERNS: List[re.Pattern] = [
    # taxonomy / systematics debates
    re.compile(
        r"(?i)(phylogenet(ic|ics)"
        r"|cladistic"
        r"|taxonom(y|ic)"
        r"|systemat(ic|ics)"
        r"|monophyl(y|etic)"
        r"|paraphyl(y|etic)"
        r"|polyphyl(y|etic)"
        r"|sister\s+(group|taxon|clade)"
        r"|molecular\s+(clock|phylogen)"
        r"|bayesian\s+(inference|analysis|tree)"
        r"|maximum\s+likelihood\s+tree"
        r"|bootstrap\s+(support|value)"
        r"|posterior\s+probabilit)"
    ),
    # habitat ecology descriptions (without study-site info)
    re.compile(
        r"(?i)(habitat\s+(type|preference|selection|use|suitability)"
        r"|home\s+range\s+(size|area|overlap)"
        r"|territory\s+(size|defense|overlap)"
        r"|canopy\s+(cover|closure|height)"
        r"|vegetation\s+(type|structure|cover|composition|survey)"
        r"|understory|understorey"
        r"|basal\s+area"
        r"|tree\s+(density|dbh|diameter)"
        r"|forest\s+type\s+(was|is|include))"
    ),
    # literature review / citation-heavy passages
    re.compile(
        r"(?i)(several\s+(studies|authors|investigators)\s+(have|found|reported|showed)"
        r"|previous(ly)?\s+(reported|described|documented|found|studied)"
        r"|has\s+been\s+(reported|documented|described)\s+by"
        r"|according\s+to\s+\w+\s*(\(\d{4}\)|\d{4})"
        r"|consistent\s+with\s+(the\s+)?(findings?|results?)\s+of"
        r"|\(\s*see\s+(also\s+)?\w+\s*(et\s+al\.?)?\s*,?\s*\d{4}\)"
        r"|reviewed\s+(by|in)\s+\w+)"
    ),
    # detailed prey identification methodology
    re.compile(
        r"(?i)(prey\s+(were\s+)?identified\s+(to|using|by|under)"
        r"|identification\s+(key|guide|manual)"
        r"|taxonomic\s+(key|identification)"
        r"|dichotomous\s+key"
        r"|stereomicroscope"
        r"|dissecting\s+microscope"
        r"|otolith\s+(identification|catalogue|reference)"
        r"|diagnostic\s+(bones?|fragments?|structures?)"
        r"|reference\s+collection"
        r"|hard\s+(parts?|remains?)\s+(were\s+)?(identified|compared|matched))"
    ),
    # morphometric / biometric measurements
    re.compile(
        r"(?i)(morphometric"
        r"|snout[\-\s]vent\s+length"
        r"|total\s+length\s+(was\s+)?measured"
        r"|body\s+(mass|weight)\s+(was\s+)?measured"
        r"|wing\s+(chord|length)\s+(was\s+)?measured"
        r"|bill\s+(length|depth|width)\s+(was\s+)?measured"
        r"|tarsus\s+length"
        r"|carapace\s+(length|width)"
        r"|standard\s+length\s+\(SL\)"
        r"|fork\s+length\s+\(FL\)"
        r"|total\s+length\s+\(TL\))"
    ),
    # statistical methods (not results)
    re.compile(
        r"(?i)(anova|ancova|manova"
        r"|chi[\-\s]?squared?\s+test"
        r"|kruskal[\-\s]wallis"
        r"|mann[\-\s]whitney"
        r"|wilcoxon"
        r"|tukey('?s)?\s+(hsd|post[\-\s]?hoc)"
        r"|bonferroni\s+correction"
        r"|generali[sz]ed\s+linear\s+(model|mixed)"
        r"|linear\s+regression\s+(was|were)\s+used"
        r"|principal\s+component\s+analysis"
        r"|canonical\s+correspondence"
        r"|multivariate\s+analysis\s+of"
        r"|permutational\s+anova"
        r"|rarefaction\s+curve"
        r"|shannon[\-\s]wiener|simpson('?s)?\s+(diversity|index))"
    ),
    # conservation / management policy (not data)
    re.compile(
        r"(?i)(conservation\s+(implication|strateg|management|priority|action)"
        r"|management\s+(implication|recommendation|strateg|plan)"
        r"|red\s+list\s+(status|categor)"
        r"|endangered\s+species\s+act"
        r"|iucn\s+(status|categor|red\s+list)"
        r"|population\s+(viability|modelling|decline|trend)"
        r"|threat\s+(status|assessment|categor))"
    ),
    # genetic / molecular methods (not diet data)
    re.compile(
        r"(?i)(pcr\s+(amplif|reaction|protocol|conditions)"
        r"|dna\s+(extract|amplif|sequenc|barcod)"
        r"|mitochondrial\s+(dna|gene|region|marker)"
        r"|microsatellite"
        r"|primer\s+(pair|sequence|set)"
        r"|gel\s+electrophoresis"
        r"|nucleotide\s+(sequence|substitution)"
        r"|genbank\s+accession)"
    ),
]

# ---------------------------------------------------------------------------
# Section headings — recognise both known-priority and drop-candidate headers
# so we can preserve them in output while filtering paragraph content.
# ---------------------------------------------------------------------------

_NUM_PREFIX = r"(?:\d{1,2}(?:\.\d{1,2})*\.?\s+)?"

_SECTION_HEADING_RE: re.Pattern = re.compile(
    r"(?i)^\s*" + _NUM_PREFIX + r"("
    r"abstract|summary"
    r"|introduction|background"
    r"|methods?|materials?\s*(?:and|&)\s*methods?"
    r"|methodology"
    r"|study\s*(?:area|site|design|region|period)"
    r"|results?|findings?"
    r"|discussion|conclusions?"
    r"|summary\s+and\s+discussion"
    r"|acknowledge?ments?"
    r"|literature\s+cited"
    r"|references?\s+cited"
    r"|references?"
    r"|bibliography"
    r"|appendix"
    r"|supplementary\s+(data|material|information)"
    r"|supporting\s+information"
    r"|conflict\s+of\s+interest"
    r"|competing\s+interests?"
    r"|author\s+contributions?"
    r"|funding(?:\s+(?:sources?|information))?"
    r"|data\s+availability"
    r"|ethics\s+(statement|declaration)"
    r"|table\s*\d"
    r")\s*[:\.\-]?\s*$",
)

# Page markers are always preserved.
_PAGE_MARKER_RE: re.Pattern = re.compile(r"^\s*\[PAGE\s+\d+\]\s*$")


# ---------------------------------------------------------------------------
# Long-document threshold — above this many chars the filter becomes stricter,
# dropping paragraphs that have *no* signal at all (neither positive nor
# negative).  This prevents very long papers from passing through unfiltered
# when most paragraphs simply lack a negative keyword.
# ---------------------------------------------------------------------------

_LONG_DOC_THRESHOLD: int = 15_000


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def filter_relevant_sections(text: str) -> str:
    """Remove paragraphs unlikely to contain target diet metrics.

    The function splits *text* into blank-line-separated paragraphs, scores
    each one against positive (keep) and negative (drop) pattern lists, and
    removes paragraphs that trigger negative patterns while scoring zero on
    positive patterns.

    For documents longer than ``_LONG_DOC_THRESHOLD`` characters, paragraphs
    with *no signal at all* (neither positive nor negative) are also dropped.
    This prevents very long papers from passing through almost unfiltered.

    ``[PAGE N]`` markers and section headings are always preserved so
    ``extract_key_sections()`` can still perform section-priority ranking
    downstream.

    Args:
        text: Noise-cleaned text (output of ``clean_text()``).  May contain
              ``[PAGE N]`` markers and section headings.

    Returns:
        Filtered text with irrelevant paragraphs removed.  All structural
        markers are preserved.
    """
    if not text or not text.strip():
        return text

    strict = len(text) > _LONG_DOC_THRESHOLD

    # Split into blocks on blank lines while tracking structure
    blocks = _split_into_blocks(text)

    kept: List[str] = []
    for block in blocks:
        stripped = block.strip()
        if not stripped:
            # Preserve blank-line spacing
            kept.append("")
            continue

        # Always keep page markers and section headings
        if _PAGE_MARKER_RE.match(stripped) or _SECTION_HEADING_RE.match(stripped):
            kept.append(block)
            continue

        # Score the paragraph
        if _should_keep(stripped, strict=strict):
            kept.append(block)

    result = "\n\n".join(kept)

    # Collapse excessive blank lines (more than 2 newlines → 2)
    result = re.sub(r"\n{3,}", "\n\n", result)

    return result.strip()


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _split_into_blocks(text: str) -> List[str]:
    """Split text into paragraph blocks separated by blank lines.

    ``[PAGE N]`` markers that appear on their own line are treated as their
    own block so they are never merged with surrounding text.
    """
    # First, ensure PAGE markers are separated by blank lines from content
    # so they end up in their own block.
    normalized = re.sub(
        r"(?<!\n)\n(\[PAGE\s+\d+\])\n(?!\n)",
        r"\n\n\1\n\n",
        text,
    )
    return re.split(r"\n\s*\n", normalized)


def _has_positive_signal(text: str) -> bool:
    """Return True if *text* matches any positive/keep pattern."""
    for pat in _POSITIVE_PATTERNS:
        if pat.search(text):
            return True
    return False


def _has_negative_signal(text: str) -> bool:
    """Return True if *text* matches any negative/drop-candidate pattern."""
    for pat in _NEGATIVE_PATTERNS:
        if pat.search(text):
            return True
    return False


def _should_keep(text: str, *, strict: bool = False) -> bool:
    """Decide whether a paragraph should be kept.

    Decision logic (conservative — defaults to keep):
      1. If the paragraph has ANY positive signal → **keep**.
      2. If the paragraph has a negative signal AND no positive signal → **drop**.
      3. If ``strict`` is True and there is NO signal either way → **drop**.
      4. Otherwise (no signal, not strict) → **keep** (borderline).

    Args:
        text: Paragraph text to evaluate.
        strict: When True (used for long documents), paragraphs with zero
            signal are dropped rather than kept by default.
    """
    if _has_positive_signal(text):
        return True
    if _has_negative_signal(text):
        return False
    # No signal either way
    if strict:
        return False  # long doc — drop borderline paragraphs
    return True
