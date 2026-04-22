from __future__ import annotations

import hashlib
import json
import math
import re
import unicodedata
import warnings
from pathlib import Path
from typing import Iterable

import pandas as pd
from charset_normalizer import from_bytes

from .config import COLUMN_ALIASES, FALSE_TOKENS, MISSING_TOKENS, TRUE_TOKENS


CAMEL_CASE_PATTERN = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")
NON_WORD_PATTERN = re.compile(r"[^a-zA-Z0-9]+")
NUMERIC_PATTERN = re.compile(r"^[+-]?\d+(\.\d+)?([eE][+-]?\d+)?$")


def ensure_directory(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def strip_accents(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value)
    return normalized.encode("ascii", "ignore").decode("ascii")


def normalize_token(value: object) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    text = clean_text(value)
    if text is pd.NA:
        return ""
    text = strip_accents(str(text)).lower()
    return re.sub(r"\s+", " ", text).strip()


def clean_text(value: object) -> object:
    if value is None or value is pd.NA:
        return pd.NA
    text = str(value).replace("\xa0", " ").strip()
    if not text:
        return pd.NA
    if normalize_token_shallow(text) in MISSING_TOKENS:
        return pd.NA
    return re.sub(r"\s+", " ", text)


def normalize_token_shallow(value: str) -> str:
    return strip_accents(value).lower().strip()


def snake_case(name: str) -> str:
    text = CAMEL_CASE_PATTERN.sub("_", str(name).strip())
    text = strip_accents(text)
    text = NON_WORD_PATTERN.sub("_", text)
    text = re.sub(r"_+", "_", text).strip("_").lower()
    return COLUMN_ALIASES.get(text, text)


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    renamed = [snake_case(column) for column in df.columns]
    df = df.copy()
    df.columns = renamed
    return df


def detect_file_type(path: Path) -> str:
    name = path.name
    if name.startswith("27_processoslicitatorios"):
        return "processos_licitatorios"
    return name.split("_", 1)[0].split("-", 1)[0]


def detect_encoding(path: Path, sample_size: int = 65536) -> str:
    with path.open("rb") as handle:
        raw = handle.read(sample_size)
    if raw.startswith(b"\xef\xbb\xbf"):
        return "utf-8-sig"
    try:
        raw.decode("utf-8")
        return "utf-8"
    except UnicodeDecodeError:
        pass
    match = from_bytes(raw).best()
    if match and match.encoding:
        return match.encoding
    return "cp1252"


def read_csv_with_detection(
    path: Path,
    *,
    nrows: int | None = None,
    usecols: list[str] | None = None,
) -> pd.DataFrame:
    encoding = detect_encoding(path)
    return pd.read_csv(
        path,
        encoding=encoding,
        encoding_errors="replace",
        dtype="string",
        keep_default_na=False,
        na_filter=False,
        nrows=nrows,
        usecols=usecols,
    )


def has_reference(value: object) -> bool:
    text = clean_text(value)
    if text is pd.NA:
        return False
    return str(text).lower().endswith(".csv")


def parse_boolean(value: object) -> object:
    token = normalize_token(value)
    if token in TRUE_TOKENS:
        return True
    if token in FALSE_TOKENS:
        return False
    return pd.NA


def parse_boolean_series(series: pd.Series) -> pd.Series:
    return series.map(parse_boolean).astype("boolean")


def boolean_to_int_series(series: pd.Series) -> pd.Series:
    mapped = series.map(parse_boolean)
    return mapped.map(lambda value: 1 if value is True else 0).astype("int8")


def normalize_string_series(series: pd.Series) -> pd.Series:
    return series.map(clean_text).astype("string")


def _normalize_numeric_string(value: object) -> str | None:
    cleaned = clean_text(value)
    if cleaned is pd.NA:
        return None
    text = str(cleaned).replace(" ", "")
    if NUMERIC_PATTERN.match(text):
        return text
    if "," in text and "." in text:
        if text.rfind(",") > text.rfind("."):
            return text.replace(".", "").replace(",", ".")
        return text.replace(",", "")
    if "," in text:
        left, right = text.rsplit(",", 1)
        if len(right) <= 2:
            return left.replace(".", "") + "." + right
        return text.replace(",", "")
    return text


def parse_numeric_series(series: pd.Series) -> pd.Series:
    normalized = series.map(_normalize_numeric_string)
    return pd.to_numeric(normalized, errors="coerce")


def parse_datetime_series(series: pd.Series) -> pd.Series:
    cleaned = series.map(clean_text)
    converted = pd.to_datetime(cleaned, errors="coerce", utc=True)
    try:
        return converted.dt.tz_convert(None)
    except AttributeError:
        return converted


def stable_hash(*parts: object, prefix: str | None = None, length: int = 16) -> str:
    raw = "||".join("" if part is None or part is pd.NA else str(part) for part in parts)
    digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:length]
    if prefix:
        return f"{prefix}_{digest}"
    return digest


def header_signature(columns: Iterable[str]) -> str:
    joined = "||".join(columns)
    return hashlib.sha1(joined.encode("utf-8")).hexdigest()


def infer_column_kinds(df: pd.DataFrame) -> dict[str, str]:
    kinds: dict[str, str] = {}
    for column in df.columns:
        series = normalize_string_series(df[column]).dropna()
        if series.empty:
            kinds[column] = "empty"
            continue
        sample = series.head(20)
        bool_values = sample.map(parse_boolean).dropna()
        if len(bool_values) == len(sample):
            kinds[column] = "boolean_like"
            continue
        numeric = pd.to_numeric(sample.map(_normalize_numeric_string), errors="coerce")
        if numeric.notna().mean() >= 0.8:
            kinds[column] = "numeric_like"
            continue
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            warnings.simplefilter("ignore", category=FutureWarning)
            datetimes = pd.to_datetime(sample, errors="coerce")
        if datetimes.notna().mean() >= 0.8:
            kinds[column] = "datetime_like"
            continue
        kinds[column] = "text"
    return kinds


def text_length(series: pd.Series) -> pd.Series:
    normalized = normalize_string_series(series)
    return normalized.fillna("").str.len().astype("Int64")


def word_count(series: pd.Series) -> pd.Series:
    normalized = normalize_string_series(series)
    counts = normalized.fillna("").str.split().map(len)
    return counts.astype("Int64")


def write_json(path: Path, payload: object) -> None:
    ensure_directory(path.parent)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def concat_frames(frames: list[pd.DataFrame]) -> pd.DataFrame:
    valid_frames = [frame for frame in frames if frame is not None and not frame.empty]
    if not valid_frames:
        return pd.DataFrame()
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            category=FutureWarning,
            message="The behavior of DataFrame concatenation with empty or all-NA entries is deprecated.*",
        )
        return pd.concat(valid_frames, ignore_index=True)
