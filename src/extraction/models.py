"""Pydantic models for predator diet data extraction.

PredatorDietMetrics is built dynamically from the FIELDS registry in
src/config.py so that adding or removing an extraction field requires only
editing that one file.
"""

import re
from typing import Callable, Optional

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    computed_field,
    create_model,
    field_validator,
    model_validator,
)

from src.config import FIELDS, FieldSpec

# ---------------------------------------------------------------------------
# Month lookup table
# ---------------------------------------------------------------------------

_MONTH_MAP = {
    "january": "01",
    "february": "02",
    "march": "03",
    "april": "04",
    "may": "05",
    "june": "06",
    "july": "07",
    "august": "08",
    "september": "09",
    "october": "10",
    "november": "11",
    "december": "12",
    "jan": "01",
    "feb": "02",
    "mar": "03",
    "apr": "04",
    "jun": "06",
    "jul": "07",
    "aug": "08",
    "sep": "09",
    "oct": "10",
    "nov": "11",
    "dec": "12",
}


# ---------------------------------------------------------------------------
# Built-in normalizer functions (module-level, no cls parameter)
# ---------------------------------------------------------------------------


def _normalize_year_range(value):
    if value is None:
        return None
    if isinstance(value, (int, float)):
        value = str(int(value))
    elif not isinstance(value, str):
        return None
    years = re.findall(r"\b(\d{4})\b", value)
    if not years:
        return None
    if len(years) >= 2 and years[0] != years[1]:
        y_sorted = sorted(years[:2], key=int)
        return f"{y_sorted[0]}-{y_sorted[1]}"
    return years[0]


def _normalize_year(value):
    if value is None:
        return None
    if isinstance(value, (int, float)):
        value = str(int(value))
    elif not isinstance(value, str):
        return None
    years = re.findall(r"\b(\d{4})\b", value)
    if not years:
        return None
    if len(years) >= 2 and years[0] != years[1]:
        y1, y2 = sorted([int(years[0]), int(years[1])])
        return str((y1 + y2 + 1) // 2)
    return years[0]


def _normalize_month(value):
    if value is None:
        return None
    if isinstance(value, (int, float)):
        n = int(value)
        if 1 <= n <= 12:
            return f"{n:02d}"
        return None
    elif not isinstance(value, str):
        return None
    v = value.strip()
    name = v.lower()
    if name in _MONTH_MAP:
        return _MONTH_MAP[name]
    if re.match(r"^\d{1,2}$", v):
        n = int(v)
        if 1 <= n <= 12:
            return f"{n:02d}"
    return None


def _normalize_day(value):
    if value is None:
        return None
    if isinstance(value, (int, float)):
        n = int(value)
        if 1 <= n <= 31:
            return f"{n:02d}"
        return None
    elif not isinstance(value, str):
        return None
    v = value.strip()
    if re.match(r"^\d{1,2}$", v):
        n = int(v)
        if 1 <= n <= 31:
            return f"{n:02d}"
    return None


_NORMALIZER_MAP: dict[str, Callable] = {
    "year_range": _normalize_year_range,
    "year": _normalize_year,
    "month": _normalize_month,
    "day": _normalize_day,
}


# ---------------------------------------------------------------------------
# Base class — cross-field validator and computed field only
# ---------------------------------------------------------------------------


class _MetricsBase(BaseModel):
    model_config = ConfigDict(
        strict=True,
        validate_default=True,
        str_strip_whitespace=True,
        frozen=False,
    )

    @model_validator(mode="after")
    def _reconcile_num_sampled(self):
        """Fill num_sampled from components when absent; warn on mismatch but keep paper value."""
        import logging as _logging

        _log = _logging.getLogger(__name__)

        empty = getattr(self, "num_empty", None)
        nonempty = getattr(self, "num_nonempty", None)

        # Both zero is logically impossible (no animal can be neither empty nor non-empty).
        # It indicates a mis-classified paper or a failed extraction; reset to null.
        if empty == 0 and nonempty == 0:
            _log.warning("num_empty=0 and num_nonempty=0 simultaneously — invalid extraction result; " "resetting both to null (paper likely mis-classified or extraction failed)")
            object.__setattr__(self, "num_empty", None)
            object.__setattr__(self, "num_nonempty", None)
            return self

        if empty is not None and nonempty is not None:
            calculated = empty + nonempty
            current = getattr(self, "num_sampled", None)
            if current is None and calculated > 0:
                object.__setattr__(self, "num_sampled", calculated)
            elif current is not None and current != calculated:
                _log.warning(
                    "num_sampled=%d != num_empty(%d) + num_nonempty(%d)=%d — keeping paper value",
                    current,
                    empty,
                    nonempty,
                    calculated,
                )
        return self

    @computed_field(description="Fraction of predators that had food in their stomachs (0.0–1.0).")
    @property
    def fraction_feeding(self) -> Optional[float]:
        nonempty = getattr(self, "num_nonempty", None)
        sampled = getattr(self, "num_sampled", None)
        if nonempty is not None and sampled is not None and sampled > 0:
            return round(nonempty / sampled, 4)
        return None


# ---------------------------------------------------------------------------
# Import-time validation of FIELDS
# ---------------------------------------------------------------------------

_VALID_TYPES = {Optional[str], Optional[int], Optional[float]}
_RESERVED = set(_MetricsBase.model_computed_fields.keys()) | {"model_config"}

for _spec in FIELDS:
    assert _spec.name.isidentifier(), f"FieldSpec: invalid name {_spec.name!r}"
    assert _spec.name not in _RESERVED, f"FieldSpec: reserved name {_spec.name!r}"
    assert _spec.python_type in _VALID_TYPES, f"FieldSpec: unsupported python_type for {_spec.name!r} — " f"must be Optional[str], Optional[int], or Optional[float]"
    if _spec.pattern:
        re.compile(_spec.pattern)
    if _spec.ge is not None and _spec.le is not None:
        assert _spec.ge <= _spec.le, f"FieldSpec: ge > le for {_spec.name!r}"


# ---------------------------------------------------------------------------
# Dynamic model construction
# ---------------------------------------------------------------------------


def _make_field_validator(field_name: str, norm_fn: Callable):
    """Return a Pydantic field_validator for field_name that applies norm_fn."""

    def _v(cls, value):
        return norm_fn(value)

    _v.__name__ = f"_validate_{field_name}"
    _v.__qualname__ = f"_validate_{field_name}"
    return field_validator(field_name, mode="before")(classmethod(_v))


_field_defs: dict = {}
_validators: dict = {}

for _spec in FIELDS:
    kwargs: dict = {"default": None, "description": _spec.description}
    if _spec.pattern is not None:
        kwargs["pattern"] = _spec.pattern
    if _spec.min_length is not None:
        kwargs["min_length"] = _spec.min_length
    if _spec.max_length is not None:
        kwargs["max_length"] = _spec.max_length
    if _spec.ge is not None:
        kwargs["ge"] = _spec.ge
    if _spec.le is not None:
        kwargs["le"] = _spec.le
    if _spec.gt is not None:
        kwargs["gt"] = _spec.gt
    _field_defs[_spec.name] = (_spec.python_type, Field(**kwargs))

    if _spec.normalizer is not None:
        if isinstance(_spec.normalizer, str):
            norm_fn = _NORMALIZER_MAP[_spec.normalizer]
        else:
            norm_fn = _spec.normalizer
        _validators[f"_validate_{_spec.name}"] = _make_field_validator(_spec.name, norm_fn)


PredatorDietMetrics = create_model(
    "PredatorDietMetrics",
    __base__=_MetricsBase,
    __validators__=_validators,
    **_field_defs,
)


class ExtractionResult(BaseModel):
    """Wrapper returned by the LLM: one record per (species, survey) pair."""

    records: list[PredatorDietMetrics]
