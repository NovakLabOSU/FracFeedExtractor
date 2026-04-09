"""Unit tests for src/llm/models.py — PredatorDietMetrics Pydantic schema."""

import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.llm.models import PredatorDietMetrics


# ---------------------------------------------------------------------------
# Synthetic diet survey data — stands in for a real document
# ---------------------------------------------------------------------------

# Mimics data extracted from a stomach-content study of Atlantic cod:
#   - 200 individuals collected, 45 had empty stomachs, 155 had food
#   - Study site: Grand Banks, Newfoundland
#   - Collection period: 1998-2000
FULL_SURVEY = dict(
    species_name="Gadus morhua",
    study_location="Grand Banks, Newfoundland, Canada",
    study_date="1998-2000",
    num_empty_stomachs=45,
    num_nonempty_stomachs=155,
    sample_size=200,
)

# All-food stomach-pumping study (num_empty always 0)
PUMP_SURVEY = dict(
    species_name="Pygoscelis papua",
    study_location="Marion Island, sub-Antarctic",
    study_date="1984-1985",
    num_empty_stomachs=0,
    num_nonempty_stomachs=144,
    sample_size=144,
)

# Minimal survey — only species and sample size known
MINIMAL_SURVEY = dict(
    species_name="Ursus arctos",
    study_location=None,
    study_date="2020",
    num_empty_stomachs=None,
    num_nonempty_stomachs=None,
    sample_size=23,
)


# ---------------------------------------------------------------------------
# Basic construction
# ---------------------------------------------------------------------------


class TestBasicConstruction:
    def test_full_survey_constructs(self):
        m = PredatorDietMetrics(**FULL_SURVEY)
        assert m.species_name == "Gadus morhua"
        assert m.study_location == "Grand Banks, Newfoundland, Canada"
        assert m.study_date == "1998-2000"
        assert m.num_empty_stomachs == 45
        assert m.num_nonempty_stomachs == 155
        assert m.sample_size == 200

    def test_all_null_optional_fields(self):
        m = PredatorDietMetrics()
        assert m.species_name is None
        assert m.study_location is None
        assert m.study_date is None
        assert m.num_empty_stomachs is None
        assert m.num_nonempty_stomachs is None
        assert m.sample_size is None

    def test_minimal_survey_constructs(self):
        m = PredatorDietMetrics(**MINIMAL_SURVEY)
        assert m.species_name == "Ursus arctos"
        assert m.sample_size == 23
        assert m.num_empty_stomachs is None
        assert m.num_nonempty_stomachs is None


# ---------------------------------------------------------------------------
# sample_size auto-reconciliation
# ---------------------------------------------------------------------------


class TestSampleSizeReconciliation:
    def test_sample_size_set_from_empty_plus_nonempty(self):
        m = PredatorDietMetrics(num_empty_stomachs=45, num_nonempty_stomachs=155, sample_size=None)
        assert m.sample_size == 200

    def test_wrong_sample_size_corrected(self):
        """Model validator should override an incorrect sample_size."""
        m = PredatorDietMetrics(num_empty_stomachs=10, num_nonempty_stomachs=90, sample_size=999)
        assert m.sample_size == 100

    def test_correct_sample_size_unchanged(self):
        m = PredatorDietMetrics(**FULL_SURVEY)
        assert m.sample_size == 200

    def test_pump_survey_zero_empty(self):
        m = PredatorDietMetrics(**PUMP_SURVEY)
        assert m.num_empty_stomachs == 0
        assert m.num_nonempty_stomachs == 144
        assert m.sample_size == 144

    def test_only_empty_provided_no_reconciliation(self):
        """Cannot reconcile without both counts — sample_size stays None."""
        m = PredatorDietMetrics(num_empty_stomachs=10, num_nonempty_stomachs=None, sample_size=None)
        assert m.sample_size is None

    def test_only_nonempty_provided_no_reconciliation(self):
        m = PredatorDietMetrics(num_empty_stomachs=None, num_nonempty_stomachs=90, sample_size=None)
        assert m.sample_size is None


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
        m = PredatorDietMetrics(num_empty_stomachs=5, num_nonempty_stomachs=None, sample_size=50)
        assert m.fraction_feeding is None

    def test_fraction_none_when_sample_size_missing(self):
        m = PredatorDietMetrics(num_nonempty_stomachs=30, sample_size=None)
        assert m.fraction_feeding is None

    def test_fraction_none_when_all_null(self):
        m = PredatorDietMetrics()
        assert m.fraction_feeding is None

    def test_fraction_rounded_to_4_decimals(self):
        # 2 empty + 1 nonempty → sample_size=3, fraction = 1/3 = 0.3333
        m = PredatorDietMetrics(num_empty_stomachs=2, num_nonempty_stomachs=1)
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
# Field-level validation — study_date
# ---------------------------------------------------------------------------


class TestStudyDateValidation:
    def test_single_year(self):
        m = PredatorDietMetrics(study_date="2019")
        assert m.study_date == "2019"

    def test_year_range_hyphen(self):
        m = PredatorDietMetrics(study_date="2015-2018")
        assert m.study_date == "2015-2018"

    def test_null_date_allowed(self):
        m = PredatorDietMetrics(study_date=None)
        assert m.study_date is None

    def test_text_date_rejected(self):
        with pytest.raises(ValidationError):
            PredatorDietMetrics(study_date="March 2019")

    def test_partial_year_rejected(self):
        with pytest.raises(ValidationError):
            PredatorDietMetrics(study_date="19")

    def test_three_year_range_rejected(self):
        with pytest.raises(ValidationError):
            PredatorDietMetrics(study_date="2015-2018-2020")


# ---------------------------------------------------------------------------
# Field-level validation — count fields
# ---------------------------------------------------------------------------


class TestCountFieldValidation:
    def test_negative_empty_stomachs_rejected(self):
        with pytest.raises(ValidationError):
            PredatorDietMetrics(num_empty_stomachs=-1)

    def test_negative_nonempty_stomachs_rejected(self):
        with pytest.raises(ValidationError):
            PredatorDietMetrics(num_nonempty_stomachs=-5)

    def test_zero_empty_stomachs_allowed(self):
        m = PredatorDietMetrics(num_empty_stomachs=0)
        assert m.num_empty_stomachs == 0

    def test_zero_sample_size_rejected(self):
        with pytest.raises(ValidationError):
            PredatorDietMetrics(sample_size=0)

    def test_negative_sample_size_rejected(self):
        with pytest.raises(ValidationError):
            PredatorDietMetrics(sample_size=-10)

    def test_large_counts_accepted(self):
        m = PredatorDietMetrics(num_empty_stomachs=0, num_nonempty_stomachs=10000, sample_size=10000)
        assert m.sample_size == 10000


# ---------------------------------------------------------------------------
# Serialization round-trip
# ---------------------------------------------------------------------------


class TestSerialization:
    def test_model_dump_round_trip(self):
        m = PredatorDietMetrics(**FULL_SURVEY)
        d = m.model_dump()
        m2 = PredatorDietMetrics.model_validate(d)
        assert m2.species_name == m.species_name
        assert m2.sample_size == m.sample_size
        assert m2.fraction_feeding == m.fraction_feeding

    def test_model_validate_json(self):
        import json

        payload = json.dumps(
            {
                "species_name": "Vulpes vulpes",
                "study_location": "Bristol, UK",
                "study_date": "2015-2018",
                "num_empty_stomachs": 12,
                "num_nonempty_stomachs": 88,
                "sample_size": 100,
            }
        )
        m = PredatorDietMetrics.model_validate_json(payload)
        assert m.species_name == "Vulpes vulpes"
        assert m.fraction_feeding == pytest.approx(0.88)

    def test_model_json_schema_contains_required_fields(self):
        schema = PredatorDietMetrics.model_json_schema()
        props = schema.get("properties", {})
        assert "species_name" in props
        assert "study_location" in props
        assert "study_date" in props
        assert "num_empty_stomachs" in props
        assert "num_nonempty_stomachs" in props
        assert "sample_size" in props
