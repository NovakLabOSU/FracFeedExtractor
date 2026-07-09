"""Unit tests for src/extraction/models.py — PredatorDietMetrics Pydantic schema."""

import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.extraction.models import ExtractionResult, PredatorDietMetrics


# ---------------------------------------------------------------------------
# Synthetic diet survey data — stands in for a real document
# ---------------------------------------------------------------------------

# Mimics data extracted from a stomach-content study of Atlantic cod:
#   - 200 individuals collected, 45 had empty stomachs, 155 had food
#   - Study site: Grand Banks, Newfoundland
#   - Collection period: 1998-2000 (midpoint year: 1999)
FULL_SURVEY = dict(
    species_name="Gadus morhua",
    study_location="Grand Banks, Newfoundland, Canada",
    study_year_range="1998-2000",
    study_year="1999",
    num_empty=45,
    num_nonempty=155,
    num_sampled=200,
)

# All-food stomach-pumping study (num_empty always 0)
PUMP_SURVEY = dict(
    species_name="Pygoscelis papua",
    study_location="Marion Island, sub-Antarctic",
    study_year_range="1984-1985",
    study_year="1985",
    num_empty=0,
    num_nonempty=144,
    num_sampled=144,
)

# Minimal survey — only species and sample size known
MINIMAL_SURVEY = dict(
    species_name="Ursus arctos",
    study_location=None,
    study_year="2020",
    num_empty=None,
    num_nonempty=None,
    num_sampled=23,
)


# ---------------------------------------------------------------------------
# Basic construction
# ---------------------------------------------------------------------------


class TestBasicConstruction:
    def test_full_survey_constructs(self):
        m = PredatorDietMetrics(**FULL_SURVEY)
        assert m.species_name == "Gadus morhua"
        assert m.study_location == "Grand Banks, Newfoundland, Canada"
        assert m.study_year_range == "1998-2000"
        assert m.study_year == "1999"
        assert m.num_empty == 45
        assert m.num_nonempty == 155
        assert m.num_sampled == 200

    def test_all_null_optional_fields(self):
        m = PredatorDietMetrics()
        assert m.species_name is None
        assert m.study_location is None
        assert m.study_year_range is None
        assert m.study_year is None
        assert m.study_month is None
        assert m.study_day is None
        assert m.num_empty is None
        assert m.num_nonempty is None
        assert m.num_sampled is None

    def test_minimal_survey_constructs(self):
        m = PredatorDietMetrics(**MINIMAL_SURVEY)
        assert m.species_name == "Ursus arctos"
        assert m.num_sampled == 23
        assert m.num_empty is None
        assert m.num_nonempty is None


# ---------------------------------------------------------------------------
# num_sampled auto-reconciliation
# ---------------------------------------------------------------------------


class TestSampleSizeReconciliation:
    def test_num_sampled_set_from_empty_plus_nonempty(self):
        m = PredatorDietMetrics(num_empty=45, num_nonempty=155, num_sampled=None)
        assert m.num_sampled == 200

    def test_inconsistent_num_sampled_kept_with_warning(self, caplog):
        """Paper's num_sampled is preserved when it doesn't match num_empty + num_nonempty; a warning is logged."""
        import logging
        with caplog.at_level(logging.WARNING, logger="src.extraction.models"):
            m = PredatorDietMetrics(num_empty=10, num_nonempty=90, num_sampled=999)
        assert m.num_sampled == 999
        assert any("keeping paper value" in r.message for r in caplog.records)

    def test_correct_num_sampled_unchanged(self):
        m = PredatorDietMetrics(**FULL_SURVEY)
        assert m.num_sampled == 200

    def test_pump_survey_zero_empty(self):
        m = PredatorDietMetrics(**PUMP_SURVEY)
        assert m.num_empty == 0
        assert m.num_nonempty == 144
        assert m.num_sampled == 144

    def test_only_empty_provided_no_reconciliation(self):
        """Cannot reconcile without both counts — num_sampled stays None."""
        m = PredatorDietMetrics(num_empty=10, num_nonempty=None, num_sampled=None)
        assert m.num_sampled is None

    def test_only_nonempty_provided_no_reconciliation(self):
        m = PredatorDietMetrics(num_empty=None, num_nonempty=90, num_sampled=None)
        assert m.num_sampled is None


# ---------------------------------------------------------------------------
# fraction_feeding computed field
# ---------------------------------------------------------------------------


class TestFractionFeeding:
    def test_fraction_computed_correctly(self):
        m = PredatorDietMetrics(**FULL_SURVEY)
        assert m.fraction_feeding == pytest.approx(0.775, abs=1e-4)

    def test_fraction_all_feeding(self):
        m = PredatorDietMetrics(**PUMP_SURVEY)
        assert m.fraction_feeding == pytest.approx(1.0)

    def test_fraction_none_when_nonempty_missing(self):
        m = PredatorDietMetrics(num_empty=5, num_nonempty=None, num_sampled=50)
        assert m.fraction_feeding is None

    def test_fraction_none_when_num_sampled_missing(self):
        m = PredatorDietMetrics(num_nonempty=30, num_sampled=None)
        assert m.fraction_feeding is None

    def test_fraction_none_when_all_null(self):
        m = PredatorDietMetrics()
        assert m.fraction_feeding is None

    def test_fraction_rounded_to_4_decimals(self):
        # 2 empty + 1 nonempty → num_sampled=3, fraction = 1/3 = 0.3333
        m = PredatorDietMetrics(num_empty=2, num_nonempty=1)
        assert m.fraction_feeding == pytest.approx(0.3333, abs=1e-4)

    def test_fraction_not_in_model_dump_base_fields(self):
        """fraction_feeding is a computed field and should appear in model_dump()."""
        m = PredatorDietMetrics(**FULL_SURVEY)
        d = m.model_dump()
        assert "fraction_feeding" in d


# ---------------------------------------------------------------------------
# Field-level validation — species_name
# ---------------------------------------------------------------------------


class TestSpeciesNameValidation:
    def test_valid_binomial(self):
        m = PredatorDietMetrics(species_name="Canis lupus")
        assert m.species_name == "Canis lupus"

    def test_valid_three_word_name(self):
        m = PredatorDietMetrics(species_name="Canis lupus familiaris")
        assert m.species_name == "Canis lupus familiaris"

    def test_null_species_allowed(self):
        m = PredatorDietMetrics(species_name=None)
        assert m.species_name is None

    def test_lowercase_genus_rejected(self):
        with pytest.raises(ValidationError):
            PredatorDietMetrics(species_name="canis lupus")

    def test_all_caps_rejected(self):
        with pytest.raises(ValidationError):
            PredatorDietMetrics(species_name="CANIS LUPUS")

    def test_too_short_rejected(self):
        with pytest.raises(ValidationError):
            PredatorDietMetrics(species_name="Ab")

    def test_whitespace_stripped(self):
        m = PredatorDietMetrics(species_name="  Vulpes vulpes  ")
        assert m.species_name == "Vulpes vulpes"


# ---------------------------------------------------------------------------
# Field-level validation — study_year_range
# ---------------------------------------------------------------------------


class TestStudyYearRangeValidation:
    def test_single_year(self):
        m = PredatorDietMetrics(study_year_range="1987")
        assert m.study_year_range == "1987"

    def test_year_range(self):
        m = PredatorDietMetrics(study_year_range="1998-2000")
        assert m.study_year_range == "1998-2000"

    def test_null_allowed(self):
        m = PredatorDietMetrics(study_year_range=None)
        assert m.study_year_range is None

    def test_free_form_range_normalized(self):
        m = PredatorDietMetrics(study_year_range="from April 1984 to March 1986")
        assert m.study_year_range == "1984-1986"

    def test_same_year_repeated_returns_single(self):
        m = PredatorDietMetrics(study_year_range="collected in 1987 and 1987")
        assert m.study_year_range == "1987"

    def test_no_year_becomes_none(self):
        m = PredatorDietMetrics(study_year_range="spring")
        assert m.study_year_range is None


# ---------------------------------------------------------------------------
# Field-level validation — study_year
# ---------------------------------------------------------------------------


class TestStudyYearValidation:
    def test_single_year(self):
        m = PredatorDietMetrics(study_year="2019")
        assert m.study_year == "2019"

    def test_range_string_yields_midpoint(self):
        # "2015-2018" → (2015+2018+1)//2 = 2017
        m = PredatorDietMetrics(study_year="2015-2018")
        assert m.study_year == "2017"

    def test_free_form_range_yields_midpoint(self):
        # "from April 1984 to March 1986" → midpoint of 1984+1986 = 1985
        m = PredatorDietMetrics(study_year="from April 1984 to March 1986")
        assert m.study_year == "1985"

    def test_adjacent_years_midpoint_rounds_up(self):
        # 1984+1985 midpoint = (1984+1985+1)//2 = 1985
        m = PredatorDietMetrics(study_year="1984-1985")
        assert m.study_year == "1985"

    def test_year_extracted_from_text(self):
        # "March 2019" → validator extracts 2019
        m = PredatorDietMetrics(study_year="March 2019")
        assert m.study_year == "2019"

    def test_no_year_becomes_none(self):
        # "19" has no 4-digit year
        m = PredatorDietMetrics(study_year="19")
        assert m.study_year is None

    def test_null_allowed(self):
        m = PredatorDietMetrics(study_year=None)
        assert m.study_year is None


# ---------------------------------------------------------------------------
# Field-level validation — study_month
# ---------------------------------------------------------------------------


class TestStudyMonthValidation:
    def test_zero_padded_string(self):
        m = PredatorDietMetrics(study_month="03")
        assert m.study_month == "03"

    def test_bare_digit_zero_padded(self):
        m = PredatorDietMetrics(study_month="3")
        assert m.study_month == "03"

    def test_month_name_converted(self):
        m = PredatorDietMetrics(study_month="March")
        assert m.study_month == "03"

    def test_month_name_case_insensitive(self):
        m = PredatorDietMetrics(study_month="march")
        assert m.study_month == "03"

    def test_abbreviated_month(self):
        m = PredatorDietMetrics(study_month="Apr")
        assert m.study_month == "04"

    def test_december(self):
        m = PredatorDietMetrics(study_month="December")
        assert m.study_month == "12"

    def test_null_allowed(self):
        m = PredatorDietMetrics(study_month=None)
        assert m.study_month is None

    def test_invalid_month_becomes_none(self):
        m = PredatorDietMetrics(study_month="spring")
        assert m.study_month is None

    def test_out_of_range_becomes_none(self):
        m = PredatorDietMetrics(study_month="13")
        assert m.study_month is None


# ---------------------------------------------------------------------------
# Field-level validation — study_day
# ---------------------------------------------------------------------------


class TestStudyDayValidation:
    def test_zero_padded_string(self):
        m = PredatorDietMetrics(study_day="05")
        assert m.study_day == "05"

    def test_bare_digit_zero_padded(self):
        m = PredatorDietMetrics(study_day="5")
        assert m.study_day == "05"

    def test_two_digit_day(self):
        m = PredatorDietMetrics(study_day="15")
        assert m.study_day == "15"

    def test_last_day_of_month(self):
        m = PredatorDietMetrics(study_day="31")
        assert m.study_day == "31"

    def test_null_allowed(self):
        m = PredatorDietMetrics(study_day=None)
        assert m.study_day is None

    def test_out_of_range_becomes_none(self):
        m = PredatorDietMetrics(study_day="32")
        assert m.study_day is None

    def test_zero_becomes_none(self):
        m = PredatorDietMetrics(study_day="0")
        assert m.study_day is None


# ---------------------------------------------------------------------------
# Field-level validation — count fields
# ---------------------------------------------------------------------------


class TestCountFieldValidation:
    def test_negative_empty_stomachs_rejected(self):
        with pytest.raises(ValidationError):
            PredatorDietMetrics(num_empty=-1)

    def test_negative_nonempty_stomachs_rejected(self):
        with pytest.raises(ValidationError):
            PredatorDietMetrics(num_nonempty=-5)

    def test_zero_empty_stomachs_allowed(self):
        m = PredatorDietMetrics(num_empty=0)
        assert m.num_empty == 0

    def test_zero_num_sampled_rejected(self):
        with pytest.raises(ValidationError):
            PredatorDietMetrics(num_sampled=0)

    def test_negative_num_sampled_rejected(self):
        with pytest.raises(ValidationError):
            PredatorDietMetrics(num_sampled=-10)

    def test_large_counts_accepted(self):
        m = PredatorDietMetrics(num_empty=0, num_nonempty=10000, num_sampled=10000)
        assert m.num_sampled == 10000


# ---------------------------------------------------------------------------
# Serialization round-trip
# ---------------------------------------------------------------------------


class TestSerialization:
    def test_model_dump_round_trip(self):
        m = PredatorDietMetrics(**FULL_SURVEY)
        d = m.model_dump()
        m2 = PredatorDietMetrics.model_validate(d)
        assert m2.species_name == m.species_name
        assert m2.study_year_range == m.study_year_range
        assert m2.study_year == m.study_year
        assert m2.num_sampled == m.num_sampled
        assert m2.fraction_feeding == m.fraction_feeding

    def test_model_validate_json(self):
        import json

        payload = json.dumps(
            {
                "species_name": "Vulpes vulpes",
                "study_location": "Bristol, UK",
                "study_year_range": "2015-2018",
                "study_year": "2017",
                "study_month": "06",
                "study_day": "15",
                "num_empty": 12,
                "num_nonempty": 88,
                "num_sampled": 100,
            }
        )
        m = PredatorDietMetrics.model_validate_json(payload)
        assert m.species_name == "Vulpes vulpes"
        assert m.study_year_range == "2015-2018"
        assert m.study_year == "2017"
        assert m.study_month == "06"
        assert m.study_day == "15"
        assert m.fraction_feeding == pytest.approx(0.88)

    def test_model_json_schema_contains_required_fields(self):
        schema = PredatorDietMetrics.model_json_schema()
        props = schema.get("properties", {})
        assert "species_name" in props
        assert "study_location" in props
        assert "study_year_range" in props
        assert "study_year" in props
        assert "study_month" in props
        assert "study_day" in props
        assert "num_empty" in props
        assert "num_nonempty" in props
        assert "num_sampled" in props


# ---------------------------------------------------------------------------
# ExtractionResult wrapper
# ---------------------------------------------------------------------------


class TestExtractionResult:
    def test_single_record(self):
        rec = PredatorDietMetrics(**FULL_SURVEY)
        result = ExtractionResult(records=[rec])
        assert len(result.records) == 1
        assert result.records[0].species_name == "Gadus morhua"

    def test_multiple_records(self):
        rec1 = PredatorDietMetrics(**FULL_SURVEY)
        rec2 = PredatorDietMetrics(**PUMP_SURVEY)
        result = ExtractionResult(records=[rec1, rec2])
        assert len(result.records) == 2
        assert result.records[0].species_name == "Gadus morhua"
        assert result.records[1].species_name == "Pygoscelis papua"

    def test_empty_records_list(self):
        result = ExtractionResult(records=[])
        assert result.records == []

    def test_json_round_trip_single(self):
        import json

        payload = json.dumps({
            "records": [{
                "species_name": "Vulpes vulpes",
                "study_location": "Bristol, UK",
                "study_year_range": "2015-2018",
                "study_year": "2017",
                "study_month": "06",
                "study_day": "15",
                "num_empty": 12,
                "num_nonempty": 88,
                "num_sampled": 100,
                "survey_type": None,
                "ecosystem": None,
                "latitude": None,
                "longitude": None,
            }]
        })
        result = ExtractionResult.model_validate_json(payload)
        assert len(result.records) == 1
        assert result.records[0].species_name == "Vulpes vulpes"
        assert result.records[0].study_day == "15"
        assert result.records[0].fraction_feeding == pytest.approx(0.88)

    def test_json_round_trip_multi_species(self):
        import json

        payload = json.dumps({
            "records": [
                {
                    "species_name": "Buteo jamaicensis",
                    "study_location": "Chihuahuan Desert, New Mexico, USA",
                    "study_year_range": "2010",
                    "study_year": "2010",
                    "num_empty": 3,
                    "num_nonempty": 45,
                    "num_sampled": 48,
                    "survey_type": "Gut content (lethal)",
                    "ecosystem": "Terrestrial",
                    "latitude": None,
                    "longitude": None,
                    "study_month": None,
                    "study_day": None,
                },
                {
                    "species_name": "Falco mexicanus",
                    "study_location": "Chihuahuan Desert, New Mexico, USA",
                    "study_year_range": "2010",
                    "study_year": "2010",
                    "num_empty": 7,
                    "num_nonempty": 31,
                    "num_sampled": 38,
                    "survey_type": "Gut content (lethal)",
                    "ecosystem": "Terrestrial",
                    "latitude": None,
                    "longitude": None,
                    "study_month": None,
                    "study_day": None,
                },
            ]
        })
        result = ExtractionResult.model_validate_json(payload)
        assert len(result.records) == 2
        species = {r.species_name for r in result.records}
        assert species == {"Buteo jamaicensis", "Falco mexicanus"}

    def test_record_validation_still_enforced(self):
        """Per-record Pydantic validators still run inside ExtractionResult."""
        import json

        payload = json.dumps({
            "records": [{
                "species_name": "canis lupus",  # invalid: lowercase genus
                "num_empty": 5,
                "num_nonempty": 45,
                "num_sampled": 50,
            }]
        })
        with pytest.raises(ValidationError):
            ExtractionResult.model_validate_json(payload)

    def test_schema_has_records_key(self):
        schema = ExtractionResult.model_json_schema()
        assert "records" in schema.get("properties", {})
        records_schema = schema["properties"]["records"]
        assert records_schema.get("type") == "array"
