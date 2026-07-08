"""Pydantic models for predator diet data extraction."""

import re
from typing import Annotated, Optional
from pydantic import BaseModel, ConfigDict, Field, NonNegativeInt, computed_field, field_validator, model_validator


class PredatorDietMetrics(BaseModel):
    """Structured schema for extracted predator diet survey metrics.

    All count fields are non-negative integers.  When both
    ``num_empty_stomachs`` and ``num_nonempty_stomachs`` are present the
    model guarantees that ``sample_size`` equals their sum.
    ``fraction_feeding`` is derived automatically from the validated counts.
    """

    model_config = ConfigDict(
        strict=True,
        validate_default=True,
        str_strip_whitespace=True,
        frozen=False,
    )

    species_name: Optional[str] = Field(
        default=None,
        min_length=3,
        max_length=200,
        pattern=r"^[A-Z][a-z]+( [a-z]+)*$",
        description=(
            "Binomial scientific name of the PRIMARY PREDATOR species studied "
            "(the animal whose stomachs were examined, not its prey). "
            "Return exactly one species. If multiple predators are studied, "
            "choose the one with the most stomach samples. "
            "Format: Capitalize genus, lowercase specific epithet (e.g., 'Canis lupus', 'Pygoscelis papua')."
        ),
        examples=["Canis lupus", "Vulpes vulpes", "Pygoscelis papua", "Ursus arctos"],
    )

    study_location: Optional[str] = Field(
        default=None,
        min_length=1,
        max_length=500,
        description=(
            "Geographic area where predator specimens were collected. "
            "Include site, region, and country if available. "
            "Common section locations: Methods, Study Area, Study Site, or Materials and Methods. "
            "Examples: 'Marion Island, sub-Antarctic', 'Yellowstone National Park, Wyoming, USA', 'Bristol, UK'."
        ),
        examples=["Yellowstone National Park, Wyoming, USA", "Marion Island, sub-Antarctic", "Bristol, UK"],
    )

    study_date: Optional[str] = Field(
        default=None,
        min_length=4,
        max_length=30,
        pattern=r"^\d{4}(-\d{4})?$",
        description=(
            "Year or year-range when specimens were COLLECTED (not publication year). "
            "Format: 'YYYY' for single year or 'YYYY-YYYY' for range (e.g., '2019' or '2019-2021'). "
            "Common phrasings: 'specimens collected in', 'sampling period', 'field season', "
            "'between [year] and [year]', 'during [year]', 'from [year] to [year]'. "
        ),
        examples=["2019", "2019-2021", "1984-1985", "2015-2018"],
    )

    num_empty_stomachs: Optional[NonNegativeInt] = Field(
        default=None,
        description=(
            "Number of predators with empty stomachs (no food present). "
            "Common phrasings: 'empty', 'vacant', 'without food', 'zero prey items', "
            "'no contents', 'stomachs with no contents', 'N individuals had empty stomachs', "
            "'N empty', 'N with no prey'. Must be >= 0."
        ),
    )

    num_nonempty_stomachs: Optional[NonNegativeInt] = Field(
        default=None,
        description=(
            "Number of predators with non-empty (food-containing) stomachs. "
            "Common phrasings: 'non-empty', 'with food', 'containing prey', 'with contents', "
            "'fed', 'N contained food', 'N had prey items', 'N with prey'. Must be >= 0."
        ),
    )

    sample_size: Optional[Annotated[int, Field(gt=0)]] = Field(
        default=None,
        description=(
            "Total number of predator individuals whose stomachs (or gut contents) were examined. "
            "Must be a positive integer (> 0). This is the count of predators dissected, stomach-pumped, "
            "or otherwise sampled — NOT the number of prey items found. "
            "Common phrasings: 'N stomachs were examined', 'a total of N individuals', 'N specimens', "
            "'n=N', 'sample size of N', 'N predators were sampled'. "
            "When both num_empty_stomachs and num_nonempty_stomachs are reported, "
            "sample_size should equal their sum."
        ),
    )

    @field_validator("study_date", mode="before")
    @classmethod
    def _normalize_study_date(cls, value):
        """Coerce free-form date strings into 'YYYY' or 'YYYY-YYYY'.

        The LLM sometimes returns chatty values like
        '2015, between August and late-' or 'from 1984 to 1985'.  Pull the
        year(s) out so one messy field does not fail validation for the whole
        record.  If no 4-digit year is found, fall back to None.
        """
        if value is None or not isinstance(value, str):
            return value
        years = re.findall(r"\b(\d{4})\b", value)
        if not years:
            return None
        if len(years) >= 2 and years[0] != years[1]:
            return f"{years[0]}-{years[1]}"
        return years[0]

    @model_validator(mode="after")
    def _reconcile_sample_size(self) -> "PredatorDietMetrics":
        """Ensure sample_size == num_empty + num_nonempty when both counts are present."""
        empty = self.num_empty_stomachs
        nonempty = self.num_nonempty_stomachs
        if empty is not None and nonempty is not None:
            calculated = empty + nonempty
            if calculated > 0 and (self.sample_size is None or self.sample_size != calculated):
                self.sample_size = calculated
        return self

    @computed_field(
        description="Fraction of predators that had food in their stomachs (0.0–1.0).",
    )
    @property
    def fraction_feeding(self) -> Optional[float]:
        if self.num_nonempty_stomachs is not None and self.sample_size is not None and self.sample_size > 0:
            return round(self.num_nonempty_stomachs / self.sample_size, 4)
        return None
