"""Project-wide configuration constants."""

import re
from dataclasses import dataclass, field as dc_field
from pathlib import Path
from typing import Callable, Optional

DEFAULT_LLM_MODEL = "qwen3:30b"

GEOCODER_USER_AGENT = "FracFeedExtractor/1.0"
GEOCODER_CACHE_PATH = str(Path(__file__).parent.parent / ".geocode_cache")


# ---------------------------------------------------------------------------
# Custom normalizers for categorical fields
# ---------------------------------------------------------------------------


def _normalize_survey_type(value):
    if value is None or not isinstance(value, str):
        return value
    v = value.strip()
    canonical = {"Gut content (lethal)", "Gut content (lavage)", "Direct observation", "Other"}
    if v in canonical:
        return v
    v_lower = v.lower()
    for c in canonical:
        if c.lower() == v_lower:
            return c
    if any(kw in v_lower for kw in ["lethal", "dissect", "gut content (lethal)"]):
        return "Gut content (lethal)"
    if any(kw in v_lower for kw in ["lavage", "pump", "gastric", "emetic"]):
        return "Gut content (lavage)"
    if re.search(r'\bdirect observ', v_lower) or any(kw in v_lower for kw in ["visual observ", "behavioral observ"]):
        return "Direct observation"
    if re.search(r'\bother\b', v_lower) or any(kw in v_lower for kw in ["scat", "fecal", "isotope", "molecular", "pellet", "indirect"]):
        return "Other"
    return None


def _normalize_ecosystem(value):
    if value is None or not isinstance(value, str):
        return value
    v = value.strip()
    canonical = {"Marine", "Terrestrial", "Lotic", "Lentic"}
    if v in canonical:
        return v
    v_lower = v.lower()
    for c in canonical:
        if c.lower() == v_lower:
            return c
    if any(kw in v_lower for kw in ["marine", "ocean", "sea", "coastal", "estuar", "intertidal", "pelagic", "subtidal"]):
        return "Marine"
    if any(kw in v_lower for kw in ["terrestrial", "forest", "grassland", "savann", "desert", "woodland", "riparian"]):
        return "Terrestrial"
    if any(kw in v_lower for kw in ["lotic", "stream", "river", "creek", "brook", "flowing"]):
        return "Lotic"
    if any(kw in v_lower for kw in ["lentic", "lake", "pond", "reservoir", "wetland", "standing"]):
        return "Lentic"
    return None


# ---------------------------------------------------------------------------
# FieldSpec — single source of truth for all extraction fields
# ---------------------------------------------------------------------------


@dataclass
class FieldSpec:
    """Configuration for one extraction field.

    Required attributes define the field's identity and prompt text.
    Optional attributes add validation constraints and retry behaviour.
    """

    # Required
    name: str
    python_type: type
    prompt_type: str
    description: str
    csv_label: str

    # Optional
    retryable: bool = True
    hint: str = ""
    normalizer: "str | Callable | None" = dc_field(default=None)
    pattern: "str | None" = dc_field(default=None)
    min_length: "int | None" = dc_field(default=None)
    max_length: "int | None" = dc_field(default=None)
    ge: "float | None" = dc_field(default=None)
    le: "float | None" = dc_field(default=None)
    gt: "float | None" = dc_field(default=None)


# ---------------------------------------------------------------------------
# FIELDS — the unified registry of all extraction fields
# ---------------------------------------------------------------------------

FIELDS: list = [
    FieldSpec(
        name="species_name",
        python_type=Optional[str],
        prompt_type="string or null",
        description=(
            'Binomial Latin name (Genus species) of the predator whose diet is described by this record. '
            'This is the animal species being studied, not its prey. '
            'Capitalize genus, lowercase epithet (e.g., "Pygoscelis papua").'
        ),
        csv_label="Predator Species",
        retryable=True,
        hint=(
            "- species_name: Look for the binomial Latin name (Genus species) "
            "in the title or abstract or introduction.\n"
        ),
        pattern=r"^[A-Z][a-z]+( [a-z]+)*$",
        min_length=3,
        max_length=200,
    ),
    FieldSpec(
        name="study_location",
        python_type=Optional[str],
        prompt_type="string or null",
        description=(
            'Geographic area where specimens were collected. Include site, region, and country '
            'if available (e.g., "Marion Island, sub-Antarctic"). '
            "Check Methods, Study Area, Study Site, and Abstract."
        ),
        csv_label="Study Location",
        retryable=True,
        hint=(
            "- study_location: Check Methods or Study Area sections for place names, "
            "islands, countries, or coordinates.\n"
        ),
        min_length=1,
        max_length=500,
    ),
    FieldSpec(
        name="latitude",
        python_type=Optional[float],
        prompt_type="float or null",
        description=(
            "Decimal-degree latitude of the study site. Extract verbatim from text ONLY "
            "(e.g. 44.5, -12.3). If not explicitly stated as a number, output null — "
            "coordinates will be resolved from study_location. Northern hemisphere = positive, Southern hemisphere = negative."
        ),
        csv_label="Latitude",
        retryable=False,
        ge=-90.0,
        le=90.0,
    ),
    FieldSpec(
        name="longitude",
        python_type=Optional[float],
        prompt_type="float or null",
        description=(
            "Decimal-degree longitude of the study site. Extract verbatim from text ONLY. "
            "Output null if not explicitly stated as a number. Western hemisphere = negative, Eastern hemisphere = positive."
        ),
        csv_label="Longitude",
        retryable=False,
        ge=-180.0,
        le=180.0,
    ),
    FieldSpec(
        name="study_year_range",
        python_type=Optional[str],
        prompt_type="string or null",
        description=(
            'Year-range of diet survey or specimen collection, NOT publication year. Format "YYYY-YYYY".\n'
            "  Where to look:\n"
            '  - "specimens collected in", "sampling period", "field season", "between [year] and [year]"\n'
            "  For example:\n"
            '  - "April 1984" → "1984"\n'
            '  - "from April 1984 to March 1986" → "1984-1985"\n'
            '  - "from December 1996 to March 1997" → "1996-1997"'
        ),
        csv_label="Study Year Range",
        retryable=True,
        hint=(
            "- study_year_range: Look for the full collection period — 'from [year] to [year]', "
            "'between [year] and [year]', 'field season [year]-[year]'. "
            "Return 'YYYY-YYYY' for a range or 'YYYY' for a single year. "
            "Distinguish collection dates from publication/submission dates.\n"
        ),
        normalizer="year_range",
        pattern=r"^\d{4}(-\d{4})?$",
        min_length=4,
        max_length=9,
    ),
    FieldSpec(
        name="study_year",
        python_type=Optional[str],
        prompt_type="string or null",
        description=(
            'Year of diet survey or specimen collection, NOT publication year. '
            'If a year-range is given, return the midpoint year. Format "YYYY".\n'
            "  Where to look:\n"
            '  - "specimens collected in", "sampling period", "field season", "between [year] and [year]"\n'
            "  For example:\n"
            '  - "April 1984" → "1984"\n'
            '  - "from April 1984 to March 1986" → "1985"\n'
            '  - "from December 1996 to March 1997" → "1997"'
        ),
        csv_label="Study Year",
        retryable=True,
        hint=(
            "- study_year: Look for phrases like 'collected in', 'sampled during', "
            "'field season', 'from [month] [year] to [month] [year]'. "
            "Return a single midpoint year ('YYYY'). "
            "If no collection date is explicit, infer from 'Received [date]' — "
            "collection is typically 1-2 years before manuscript submission.\n"
        ),
        normalizer="year",
        pattern=r"^\d{4}$",
        min_length=4,
        max_length=4,
    ),
    FieldSpec(
        name="study_month",
        python_type=Optional[str],
        prompt_type="string or null",
        description=(
            'Month of diet survey or specimen collection, NOT publication month. '
            'If a month-range is given, return the midpoint month only if the range spans less than 6 months.'
            'Format "MM" (e.g., "03" for March).\n'
            "  Where to look:\n"
            '  - "specimens collected in", "sampling period", "field season", "between [month] and [month]"\n'
            "  For example:\n"
            '  - "March 1984" → "03"\n'
            '  - "from March 1984 to May 1984" → "04"\n'
            '  - "from March 1984 to May 1985" → null'
        ),
        csv_label="Study Month",
        retryable=False,
        normalizer="month",
        pattern=r"^\d{2}$",
        min_length=2,
        max_length=2,
    ),
    FieldSpec(
        name="study_day",
        python_type=Optional[str],
        prompt_type="string or null",
        description=(
            'Day of diet survey or specimen collection, NOT publication day. '
            'If a day-range is given, return the midpoint day only if the range spans less than 1 month. '
            'Format "DD" (e.g., "05" for March 5th).\n'
            "  Where to look:\n"
            '  - "specimens collected in", "sampling period", "field season", "between [month] and [month]"\n'
            "  For example:\n"
            '  - "March 5th 1984" → "05"\n'
            '  - "from March 5th 1984 to March 9th 1984" → "07"\n'
            '  - "from March 5th 1984 to April 9th 1984" → null'
        ),
        csv_label="Study Day",
        retryable=False,
        normalizer="day",
        pattern=r"^\d{2}$",
        min_length=2,
        max_length=2,
    ),
    FieldSpec(
        name="num_empty",
        python_type=Optional[int],
        prompt_type="integer (>= 0) or null",
        description=(
            "Number of surveyed predator individuals that were not feeding or were with NO food. "
            "Apply broadly across study methods.\n"
            '  - Stomach / Gut dissection: "empty", "vacuous","vacant", "without food", "zero prey items"\n'
            '  - Stomach pumping / gastric lavage: "yielded no food", "no contents obtained", "produced no material"\n'
            '  - Direct observation: "no prey items observed", "no food in stomachs", "not feeding", '
            '"not consuming prey", "not eating", "not foraging"\n'
            "  If ALL predators were eating or had guts that contained food, set this to 0."
        ),
        csv_label="Empty Stomachs",
        retryable=True,
        hint=(
            "- num_empty: Look for 'empty', 'no food', 'no contents', 'negative for prey'. "
            "If ALL samples had food (e.g., stomach pumping "
            "where every sample produced material), return 0.\n"
        ),
        ge=0.0,
    ),
    FieldSpec(
        name="num_nonempty",
        python_type=Optional[int],
        prompt_type="integer (>= 0) or null",
        description=(
            "Number of surveyed predator individuals that were feeding or were WITH food. "
            "Similar method mapping as above:\n"
            '  - Stomach / Gut dissection: "non-empty", "with food", "containing prey", "with contents", "feeding"\n'
            '  - Stomach pumping / gastric lavage: "food samples collected", "samples containing prey"\n'
            '  If study says "a total of N food samples was collected" and it implies that ALL samples '
            "had food, set num_nonempty = num_sampled."
        ),
        csv_label="Non-empty Stomachs",
        retryable=True,
        hint=(
            "- num_nonempty: Look for 'contained food', 'with prey', "
            "'non-empty', 'food samples collected'. If ALL samples had food, "
            "this equals num_sampled.\n"
        ),
        ge=0.0,
    ),
    FieldSpec(
        name="num_sampled",
        python_type=Optional[int],
        prompt_type="integer (> 0) or null",
        description=(
            "Total number of predator individuals examined. Equals num_empty + num_nonempty when both are known.\n"
            "  Where to look:\n"
            "  - Check Abstract, Methods, and Results.\n"
            "  Examples of phrases that indicate num_sampled:\n"
            '  - "N stomachs examined", "N individuals", "N specimens", "n=N", '
            '"a total of N specimens", "a total of N stomachs", "N samples were collected for diet", '
            '"N animals were examined"\n'
            "  If samples are reported as subgroups (e.g., 'two groups of 225'), sum them (e.g., 225 + 225 = 450)."
        ),
        csv_label="Total Stomachs",
        retryable=True,
        hint=(
            "- num_sampled: Look for 'N stomachs', 'N specimens', 'a total of N', "
            "'n=N', 'N individuals examined', 'two groups of N'. Check Abstract, "
            "Methods, and Results.\n"
        ),
        gt=0.0,
    ),
    FieldSpec(
        name="survey_type",
        python_type=Optional[str],
        prompt_type="string or null",
        description=(
            "The primary method used to assess diet. "
            "Use EXACTLY one of the following four values (or null if the method cannot be determined):\n"
            '  - "Gut content (lethal)": The predator was killed and its gastrointestinal tract was dissected or examined.\n'
            '  - "Gut content (lavage)": Stomach contents were flushed or pumped out non-lethally '
            "(e.g., gastric lavage, emetics, stomach pumping, oesophageal flushing).\n"
            '  - "Direct observation": Diet was assessed by directly observing predation events or '
            "prey items, without specimen collection (e.g., behavioral observation).\n"
            '  - "Other": Any other method when method cannot be inferred.'
        ),
        csv_label="Survey Type",
        retryable=True,
        hint=(
            "- survey_type: Look in the 'Methods' or 'Study Design' section. "
            "Dissection/killing = 'Gut content (lethal)'; stomach pump/lavage/emetic = 'Gut content (lavage)'; "
            "watching predators eat = 'Direct observation'; scat/feces/isotopes = 'Other'.\n"
        ),
        normalizer=_normalize_survey_type,
        pattern=r"^(Gut content \(lethal\)|Gut content \(lavage\)|Direct observation|Other)$",
        min_length=5,
        max_length=50,
    ),
    FieldSpec(
        name="ecosystem",
        python_type=Optional[str],
        prompt_type="string or null",
        description=(
            "The primary type of ecosystem where the diet survey was conducted. "
            "Use EXACTLY one of the following values (or null if it cannot be determined):\n"
            '  - "Marine": Saltwater environments (oceans, seas, estuaries, coastal waters, intertidal zones).\n'
            '  - "Terrestrial": Land-based environments (forests, grasslands, deserts, savannas, agricultural areas).\n'
            '  - "Lotic": Flowing freshwater environments (rivers, streams, creeks, brooks).\n'
            '  - "Lentic": Standing freshwater environments (lakes, ponds, reservoirs, wetlands).'
        ),
        csv_label="Ecosystem",
        retryable=True,
        hint=(
            "- ecosystem: Look for study area descriptions. "
            "Ocean/sea/estuary = 'Marine'; rivers/streams = 'Lotic'; "
            "lakes/ponds = 'Lentic'; land environments = 'Terrestrial'.\n"
        ),
        normalizer=_normalize_ecosystem,
        pattern=r"^(Marine|Terrestrial|Lotic|Lentic)$",
        min_length=5,
        max_length=20,
    ),
]


# ---------------------------------------------------------------------------
# Prompt assembly
# ---------------------------------------------------------------------------

_PROMPT_HEADER = (
    "You are a scientific data extraction assistant. Your task is to read a predator diet study "
    'and return a JSON object with a single key "records" whose value is an array. '
    "Each element of the array represents ONE predator species in ONE survey "
    "(a unique sampling location and/or time period), with exactly these fields:"
)

_PROMPT_RULES = """\
RULES
- Do not invent data; use null only if truly ambiguous or missing.
- If ALL samples had food, set num_empty = 0 and num_nonempty = num_sampled.
- Do not infer values on the basis of scat samples, fecal analysis, stable isotope analysis, molecular detection, pellet analysis, or immunoassays. Return null for all fields for these studies unless they also report on direct stomach/gut dissection, stomach pumping, or direct observation of feeding.
- Restrict to diet studies of predators (e.g., carnivores, piscivores, insectivores, omnivores) that consume other animals. Do not extract from studies of animals that primarily consume plants or fruit (i.e., herbivores or frugivores). Return an empty records array if the study is not about a predator diet.
- When num_empty or num_nonempty values are given as a percentage or proportion, convert to absolute numbers only when num_sampled is given.
- Ignore page markers [PAGE N].
- Prioritize Abstract, Methods, and Results sections.
- Carefully distinguish collection dates from publication/submission dates.

RECORD SPLITTING RULES
- The unit of one record is ONE predator species in ONE survey.
- A "survey" is a distinct sampling event with a unique location and/or time period (e.g., "summer 2010 at Site A" vs. "winter 2011 at Site B", or "depth 100-200 m" vs. "depth 200-300 m").
- Always create one record per predator species studied. A paper studying 3 species produces at least 3 records.
- Within each species, create one record per distinct survey ONLY when num_sampled, num_empty, or num_nonempty is separately reported for that survey. If a species's counts are given only in aggregate across surveys, use ONE record for that species with the aggregate counts.
- When BOTH multiple species AND multiple surveys are present, apply both rules independently. Example: 2 species x 2 sites with per-(species, site) counts -> 4 records. Example: 2 species x 2 sites with counts per species but not per site -> 2 records.
- Do not create a record for a species if its counts cannot be determined.
- A paper with one species and one survey returns an array with exactly one record.
- Always return {"records": [...]}; never return a bare JSON object or a bare array.

WITHIN-SPECIES SUBCATEGORY AGGREGATION
- Some studies report counts broken down within a species and survey.  For example, they may include subcategories of age class (e.g., adults, juveniles, age-0) or sex (e.g., males, females, unsexed/unknown). Always aggregate these subcategories into a single species-survey record: sum num_sampled, num_empty, and num_nonempty across all subcategories. Do not create separate records for subcategories."""

_PROMPT_EXAMPLES = """\
EXAMPLES

1. Single species, single survey — array with one record:
{"records": [{"species_name": "Canis lupus", "study_location": "Yellowstone National Park, Wyoming, USA", "latitude": null, "longitude": null, "study_year_range": "2018-2020", "study_year": "2019", "study_month": "04", "study_day": "06", "num_empty": 5, "num_nonempty": 47, "num_sampled": 52, "survey_type": "Gut content (lethal)", "ecosystem": "Terrestrial"}]}

2. Two species studied at the same location and time — array with two records (one per species):
{"records": [{"species_name": "Buteo jamaicensis", "study_location": "Chihuahuan Desert, New Mexico, USA", "latitude": null, "longitude": null, "study_year_range": "2010", "study_year": "2010", "study_month": null, "study_day": null, "num_empty": 3, "num_nonempty": 45, "num_sampled": 48, "survey_type": "Gut content (lethal)", "ecosystem": "Terrestrial"}, {"species_name": "Falco mexicanus", "study_location": "Chihuahuan Desert, New Mexico, USA", "latitude": null, "longitude": null, "study_year_range": "2010", "study_year": "2010", "study_month": null, "study_day": null, "num_empty": 7, "num_nonempty": 31, "num_sampled": 38, "survey_type": "Gut content (lethal)", "ecosystem": "Terrestrial"}]}

3. One species surveyed at two sites with separate per-site counts — array with two records (one per survey):
{"records": [{"species_name": "Pygoscelis papua", "study_location": "Marion Island, sub-Antarctic", "latitude": null, "longitude": null, "study_year_range": "1987", "study_year": "1987", "study_month": null, "study_day": null, "num_empty": 0, "num_nonempty": 80, "num_sampled": 80, "survey_type": "Gut content (lavage)", "ecosystem": "Marine"}, {"species_name": "Pygoscelis papua", "study_location": "Bouvet Island, sub-Antarctic", "latitude": null, "longitude": null, "study_year_range": "1987", "study_year": "1987", "study_month": null, "study_day": null, "num_empty": 0, "num_nonempty": 64, "num_sampled": 64, "survey_type": "Gut content (lavage)", "ecosystem": "Marine"}]}

4. One species, multiple seasonal surveys but counts reported only in aggregate — array with one record using the aggregate total:
{"records": [{"species_name": "Ardea herodias", "study_location": "Chesapeake Bay watershed, Maryland, USA", "latitude": null, "longitude": null, "study_year_range": "2005-2007", "study_year": "2006", "study_month": null, "study_day": null, "num_empty": 12, "num_nonempty": 88, "num_sampled": 100, "survey_type": "Direct observation", "ecosystem": "Lentic"}]}

5. One species, one survey, counts reported separately for adults and juveniles — aggregate into one record (60 + 40 = 100 sampled, 8 + 4 = 12 empty, 52 + 36 = 88 non-empty):
{"records": [{"species_name": "Ardea herodias", "study_location": "Chesapeake Bay watershed, Maryland, USA", "latitude": null, "longitude": null, "study_year_range": "2005-2007", "study_year": "2006", "study_month": null, "study_day": null, "num_empty": 12, "num_nonempty": 88, "num_sampled": 100, "survey_type": "Direct observation", "ecosystem": "Lentic"}]}"""


def build_prompt(text: str) -> str:
    """Assemble the full LLM extraction prompt from FIELDS and static sections."""
    field_list = "\n".join(f"  {f.name:<22} - {f.prompt_type}" for f in FIELDS)
    definitions = "\n\n".join(f"{f.name}: {f.description}" for f in FIELDS)
    return (
        _PROMPT_HEADER + "\n\n"
        + field_list + "\n\n"
        + "Use null ONLY when the value truly cannot be determined from any part of the text.\n\n"
        + "FIELD DEFINITIONS\n\n"
        + definitions + "\n\n"
        + _PROMPT_RULES + "\n\n"
        + _PROMPT_EXAMPLES
        + f"\n\nTEXT\n{text}"
    )
