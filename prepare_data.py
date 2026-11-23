"""Clean the full US-Accidents CSV, save it, THEN sample the cleaned records.

The complete cleaned CSV is kept locally under Downloads; the small compressed
sample plus audit are saved to data/ for Streamlit/GitHub. No synthetic accident
records are generated. Read and write in chunks to avoid loading millions of
rows at once. Exact duplicate-ID detection uses an in-memory set (~1 GB RAM
for a dataset of this size); this is separate from the chunk DataFrames.
"""
from __future__ import annotations

import argparse
from collections import Counter
import gzip
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
OUTPUT = DATA_DIR / "us_accidents_sample.csv.gz"
AUDIT = DATA_DIR / "preparation_summary.json"
DEFAULT_RAW = Path.home() / "Downloads" / "US_Accidents_Kaggle" / "US_Accidents_March23.csv"
DEFAULT_CLEANED = Path.home() / "Downloads" / "US_Accidents_Kaggle" / "US_Accidents_March23_CLEANED.csv.gz"
DEFAULT_CLEAN_AUDIT = Path.home() / "Downloads" / "US_Accidents_Kaggle" / "full_cleaning_summary.json"
PIPELINE_VERSION = "clean-full-then-sample-v1"

SOURCE = "https://www.kaggle.com/datasets/sobhanmoosavi/us-accidents"
MIRROR = "https://huggingface.co/datasets/yuvidhepe/us-accidents-updated"
API = "https://datasets-server.huggingface.co/rows"
MIRROR_NAME = "yuvidhepe/us-accidents-updated"
EXPECTED_TOTAL = 7_728_394
SEED = 20261105

US_STATE_CODES = frozenset((
    "AL AK AZ AR CA CO CT DE FL GA HI ID IL IN IA KS KY LA ME MD MA MI MN MS "
    "MO MT NE NV NH NJ NM NY NC ND OH OK OR PA RI SC SD TN TX UT VT VA WA WV WI WY DC"
).split())
COLS = [
    "ID", "State", "Severity", "Start_Time", "Start_Lat", "Start_Lng",
    "Weather_Condition", "Visibility(mi)", "Temperature(F)", "Wind_Speed(mph)",
]
OUTPUT_COLS = COLS[1:] + ["Year", "Month", "Month_Start"]
FULL_CLEAN_COLS = ["ID"] + OUTPUT_COLS
REQUIRED = ["ID", "State", "Severity", "Start_Time", "Start_Lat", "Start_Lng"]
WEATHER_BOUNDS = {
    "Visibility(mi)": (0.0, 100.0),
    "Temperature(F)": (-100.0, 150.0),
    "Wind_Speed(mph)": (0.0, 250.0),
}


def clean(raw: pd.DataFrame, *, keep_id=False):
    """Clean/validate one raw chunk. Do not sample or deduplicate here."""
    absent = sorted(set(REQUIRED) - set(raw.columns))
    if absent:
        raise ValueError(f"Required source columns missing: {absent}")
    df = raw.copy()
    for col in COLS:
        if col not in df:
            df[col] = np.nan
    df = df[COLS]
    n_input = len(df)
    missing = {c: int(df[c].isna().sum()) for c in COLS}

    ids = df["ID"].astype("string").str.strip()
    invalid_id = ids.isna() | ids.str.lower().isin(["", "nan", "none", "null"])
    df["ID"] = ids.mask(invalid_id)
    df["State"] = df["State"].astype("string").str.strip().str.upper()
    df["Start_Time"] = pd.to_datetime(
        df["Start_Time"].astype("string").str.slice(0, 19),
        errors="coerce", format="%Y-%m-%d %H:%M:%S",
    )
    for col in ("Severity", "Start_Lat", "Start_Lng", *WEATHER_BOUNDS):
        df[col] = pd.to_numeric(df[col], errors="coerce")

    valid_id = df["ID"].notna()
    valid_state = df["State"].isin(US_STATE_CODES).fillna(False)
    valid_severity = df["Severity"].isin([1, 2, 3, 4])
    valid_time = df["Start_Time"].between("2016-02-01", "2023-03-31 23:59:59").fillna(False)
    valid_coordinates = (
        df["Start_Lat"].between(18, 72) & df["Start_Lng"].between(-180, -65)
    ).fillna(False)
    valid = valid_id & valid_state & valid_severity & valid_time & valid_coordinates
    reasons = {
        "invalid_id": int((~valid_id).sum()),
        "invalid_state": int((~valid_state).sum()),
        "invalid_severity": int((~valid_severity).sum()),
        "invalid_start_time": int((~valid_time).sum()),
        "invalid_coordinates": int((~valid_coordinates).sum()),
    }
    df = df.loc[valid].copy()

    outliers = {}
    for col, (low, high) in WEATHER_BOUNDS.items():
        implausible = df[col].notna() & ~df[col].between(low, high)
        outliers[col] = int(implausible.sum())
        df.loc[implausible, col] = np.nan
        df[col] = df[col].astype("float32")
    df["Severity"] = df["Severity"].astype("int8")
    df["Start_Lat"] = df["Start_Lat"].astype("float32")
    df["Start_Lng"] = df["Start_Lng"].astype("float32")
    weather = df["Weather_Condition"].astype("string").str.strip()
    missing_weather = weather.isna() | weather.str.lower().isin(["", "nan", "none", "null"])
    df["Weather_Condition"] = weather.mask(missing_weather, "Unknown").fillna("Unknown")
    df["Year"] = df["Start_Time"].dt.year.astype("int16")
    df["Month"] = df["Start_Time"].dt.month.astype("int8")
    df["Month_Start"] = df["Start_Time"].dt.to_period("M").dt.to_timestamp()

    stats = {
        "input_rows": n_input,
        "removed_missing_or_invalid_required": n_input - len(df),
        "invalid_required_field_reasons_overlapping": reasons,
        "missing_counts_before_cleaning": missing,
        "implausible_optional_weather_values_set_missing": outliers,
        "optional_numeric_missing_after_cleaning": {c: int(df[c].isna().sum()) for c in WEATHER_BOUNDS},
        "weather_unknown_after_cleaning": int(df["Weather_Condition"].eq("Unknown").sum()),
        "valid_rows_before_global_deduplication": len(df),
    }
    return df[FULL_CLEAN_COLS if keep_id else OUTPUT_COLS], stats


def _add_counts(counter: Counter, series: pd.Series):
    counter.update({str(k): int(v) for k, v in series.dropna().value_counts().items()})


def _proportions(counter: Counter):
    n = sum(counter.values())
    return {k: round(v / n, 9) for k, v in sorted(counter.items())} if n else {}


def _max_abs_diff_pp(a: Counter, b: Counter):
    first, second = _proportions(a), _proportions(b)
    return round(max((abs(first.get(k, 0) - second.get(k, 0)) * 100
                      for k in set(first) | set(second)), default=0), 3)


def _input_signature(path: Path):
    stat = path.stat()
    return {"path": str(path.resolve()), "size_bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def clean_full_source(csv_path: Path, cleaned_path: Path = DEFAULT_CLEANED,
                      audit_path: Path = DEFAULT_CLEAN_AUDIT, chunk_size=150_000,
                      reuse_if_current=True):
    """FIRST PASS: clean ALL source records and SAVE a complete, deduplicated CSV.

    Returns an audit of the full-source cleaning stage. Existing cleaned output
    is reused only when its audit matches the unchanged source and pipeline.
    """
    csv_path, cleaned_path, audit_path = map(Path, (csv_path, cleaned_path, audit_path))
    if not csv_path.is_file():
        raise FileNotFoundError(csv_path)
    fingerprint = _input_signature(csv_path)
    if reuse_if_current and cleaned_path.is_file() and cleaned_path.stat().st_size > 0 and audit_path.is_file():
        try:
            cached = json.loads(audit_path.read_text(encoding="utf-8"))
            if (cached.get("pipeline_version") == PIPELINE_VERSION
                and cached.get("source_fingerprint") == fingerprint
                and cached.get("cleaned_output_path") == str(cleaned_path.resolve())
                and cached.get("cleaned_output_size_bytes") == cleaned_path.stat().st_size
                and cached.get("status") == "complete"):
                return cached
        except (OSError, ValueError, TypeError):
            pass

    cleaned_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = cleaned_path.with_name(cleaned_path.name + ".partial")
    seen_ids = set()  # exact global deduplication, including across CSV chunks
    n_raw = n_before_dedup = n_clean = n_duplicates = n_excluded = 0
    missing_counts, invalid_reasons, weather_outliers, optional_missing = Counter(), Counter(), Counter(), Counter()
    unknown_weather = 0
    full_counts = {k: Counter() for k in ("State", "Severity", "Year")}
    first_date = last_date = None
    try:
        with gzip.open(temporary, "wt", encoding="utf-8", newline="", compresslevel=5) as handle:
            first_chunk = True
            for raw in pd.read_csv(csv_path, usecols=lambda c: c in COLS,
                                   chunksize=chunk_size, low_memory=False):
                n_raw += len(raw)
                frame, stats = clean(raw, keep_id=True)
                n_excluded += stats["removed_missing_or_invalid_required"]
                n_before_dedup += len(frame)
                missing_counts.update(stats["missing_counts_before_cleaning"])
                invalid_reasons.update(stats["invalid_required_field_reasons_overlapping"])
                weather_outliers.update(stats["implausible_optional_weather_values_set_missing"])

                # Remove repeated IDs in this chunk and any earlier chunk.
                # Iterate in order so the first occurrence is kept deterministically.
                keep = []
                for value in frame["ID"].to_numpy():
                    if value in seen_ids:
                        keep.append(False)
                        n_duplicates += 1
                    else:
                        seen_ids.add(value)
                        keep.append(True)
                if frame.empty:
                    continue
                frame = frame.loc[keep]
                if frame.empty:
                    continue
                n_clean += len(frame)
                optional_missing.update({col: int(frame[col].isna().sum()) for col in WEATHER_BOUNDS})
                unknown_weather += int(frame["Weather_Condition"].eq("Unknown").sum())
                for k in full_counts:
                    _add_counts(full_counts[k], frame[k])
                dmin, dmax = frame["Start_Time"].min(), frame["Start_Time"].max()
                first_date = dmin if first_date is None else min(first_date, dmin)
                last_date = dmax if last_date is None else max(last_date, dmax)
                frame.to_csv(handle, index=False, header=first_chunk,
                             float_format="%.5f", date_format="%Y-%m-%d %H:%M:%S")
                first_chunk = False
        if n_clean == 0:
            raise RuntimeError("No rows remain after full-source cleaning.")
        if n_raw != n_excluded + n_duplicates + n_clean:
            raise AssertionError("Cleaning row counts do not reconcile.")
        temporary.replace(cleaned_path)
    finally:
        if temporary.exists():
            temporary.unlink()

    results = {
        "status": "complete", "pipeline_version": PIPELINE_VERSION,
        "source_url": SOURCE, "source_fingerprint": fingerprint,
        "raw_rows_inspected": n_raw,
        "removed_missing_or_invalid_required": n_excluded,
        "invalid_required_field_reasons_overlapping": dict(invalid_reasons),
        "duplicate_ids_removed_from_full_source": n_duplicates,
        "valid_rows_before_global_deduplication": n_before_dedup,
        "cleaned_rows": n_clean,
        "full_source_missing_counts_before_cleaning": dict(missing_counts),
        "implausible_optional_weather_values_set_missing": dict(weather_outliers),
        "optional_numeric_missing_after_cleaning": dict(optional_missing),
        "weather_unknown_after_cleaning": unknown_weather,
        "cleaned_distribution_counts": {key: dict(sorted(value.items())) for key, value in full_counts.items()},
        "first_observed_start_time": first_date.isoformat() if first_date is not None else None,
        "last_observed_start_time": last_date.isoformat() if last_date is not None else None,
        "cleaned_output_path": str(cleaned_path.resolve()),
        "cleaned_output_size_bytes": cleaned_path.stat().st_size,
        "cleaned_file_contains_id": True,
        "deduplication_method": "Exact unique-ID tracking across all chunks; retains first occurrence",
    }
    audit_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    return results


def sample_cleaned_source(cleaned_path: Path, *, fraction=0.02, chunk_size=150_000):
    """SECOND PASS: sample ONLY rows already saved in the cleaned CSV."""
    cleaned_path = Path(cleaned_path)
    if not (0 < fraction <= 1):
        raise ValueError("Sample fraction must be greater than zero and at most one.")
    if not cleaned_path.is_file():
        raise FileNotFoundError(cleaned_path)
    all_parts = []
    total_cleaned = selected = 0
    full_counts = {k: Counter() for k in ("State", "Severity", "Year")}
    sample_counts = {k: Counter() for k in ("State", "Severity", "Year")}
    for chunk in pd.read_csv(cleaned_path, chunksize=chunk_size, low_memory=False):
        if not set(FULL_CLEAN_COLS).issubset(chunk.columns):
            raise ValueError("Full cleaned CSV lacks required columns (including ID).")
        total_cleaned += len(chunk)
        hashed = pd.util.hash_pandas_object(chunk["ID"].astype("string"), index=False).to_numpy(dtype=np.uint64)
        keep = (hashed % 1_000_000) < round(fraction * 1_000_000)
        sample = chunk.loc[keep]
        selected += len(sample)
        for col in full_counts:
            _add_counts(full_counts[col], chunk[col])
            _add_counts(sample_counts[col], sample[col])
        if not sample.empty:
            all_parts.append(sample[OUTPUT_COLS].copy())
    if not all_parts:
        raise RuntimeError("No records were sampled from the cleaned source; increase fraction.")
    result = pd.concat(all_parts, ignore_index=True)
    assert len(result) == selected
    comparisons = {
        col: {
            "full_cleaned_counts": dict(sorted(full_counts[col].items())),
            "sample_counts": dict(sorted(sample_counts[col].items())),
            "max_absolute_percentage_point_difference": _max_abs_diff_pp(full_counts[col], sample_counts[col]),
        }
        for col in full_counts
    }
    report = {
        "sampling_method": "Deterministic ID-hash sampling of the persisted, fully cleaned dataset",
        "sampled_from": "saved_full_cleaned_csv",
        "fraction_target": fraction,
        "fraction_observed_of_cleaned": round(selected / total_cleaned, 8),
        "cleaned_rows_scanned_for_sampling": total_cleaned,
        "selected_from_cleaned": selected,
        "distribution_comparison": comparisons,
        "distribution_comparison_note": (
            "All distributions compare the fully cleaned original records with the final sample. "
            "These one-dimensional checks do not guarantee coverage of rare state/year/severity combinations."
        ),
    }
    return result, report


def save_sampled_dataset(sample_df: pd.DataFrame, full_cleaning_audit: dict,
                         sampling_audit: dict, *, output=OUTPUT, audit_path=AUDIT):
    """Save deployment data and a small, portable audit without local full cleaned rows."""
    output, audit_path = Path(output), Path(audit_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    sample_df.to_csv(output, index=False, compression="gzip",
                     float_format="%.5f", date_format="%Y-%m-%d %H:%M:%S")
    audit = {
        "status": "prepared", "pipeline_version": PIPELINE_VERSION, "data_source": SOURCE,
        "preparation_order": ["clean_full_source", "save_full_cleaned_csv", "sample_cleaned_csv", "save_deployment_sample"],
        "cleaning": {
            "raw_rows_inspected": full_cleaning_audit["raw_rows_inspected"],
            "removed_missing_or_invalid_required": full_cleaning_audit["removed_missing_or_invalid_required"],
            "invalid_required_field_reasons_overlapping": full_cleaning_audit["invalid_required_field_reasons_overlapping"],
            "duplicate_ids_removed_from_full_source": full_cleaning_audit["duplicate_ids_removed_from_full_source"],
            "cleaned_rows": full_cleaning_audit["cleaned_rows"],
            "full_source_missing_counts_before_cleaning": full_cleaning_audit["full_source_missing_counts_before_cleaning"],
            "implausible_optional_weather_values_set_missing": full_cleaning_audit["implausible_optional_weather_values_set_missing"],
            "optional_numeric_missing_after_cleaning": full_cleaning_audit["optional_numeric_missing_after_cleaning"],
            "weather_unknown_after_cleaning": full_cleaning_audit["weather_unknown_after_cleaning"],
            "first_observed_start_time": full_cleaning_audit["first_observed_start_time"],
            "last_observed_start_time": full_cleaning_audit["last_observed_start_time"],
        },
        "sampling": sampling_audit,
        "output_file": output.name,
        "output_rows": len(sample_df),
        "output_states": int(sample_df["State"].nunique()),
        "output_year_min": int(sample_df["Year"].min()),
        "output_year_max": int(sample_df["Year"].max()),
        "severity_counts": {str(k): int(v) for k, v in sample_df["Severity"].value_counts().sort_index().items()},
        "local_full_cleaned_csv_location": "Downloads/US_Accidents_Kaggle (kept locally, not deployed)",
    }
    audit_path.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    return audit


def prepare(mode="local", csv_path=None, fraction=0.02, output=OUTPUT, cleaned_path=DEFAULT_CLEANED,
            clean_audit_path=DEFAULT_CLEAN_AUDIT, reuse_if_current=True, audit_path=AUDIT):
    """Convenience wrapper with enforced ordering: full clean -> save -> sample."""
    if mode != "local":
        raise ValueError("This submission workflow requires the original Kaggle CSV (mode='local').")
    full_audit = clean_full_source(Path(csv_path or DEFAULT_RAW), Path(cleaned_path),
                                   Path(clean_audit_path), reuse_if_current=reuse_if_current)
    sampled, sample_audit = sample_cleaned_source(Path(cleaned_path), fraction=fraction)
    if sample_audit["cleaned_rows_scanned_for_sampling"] != full_audit["cleaned_rows"]:
        raise AssertionError("Saved cleaned-file row count differs from full-cleaning audit.")
    return save_sampled_dataset(sampled, full_audit, sample_audit, output=output, audit_path=audit_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_RAW, help="Path to official Kaggle CSV")
    parser.add_argument("--cleaned", type=Path, default=DEFAULT_CLEANED, help="Local path to save complete cleaned CSV.gz")
    parser.add_argument("--fraction", type=float, default=0.02, help="Fraction of CLEANED records to sample")
    parser.add_argument("--rebuild-cleaned", action="store_true", help="Force cleaning the full CSV again")
    args = parser.parse_args()
    result = prepare("local", csv_path=args.input, cleaned_path=args.cleaned,
                     fraction=args.fraction, reuse_if_current=not args.rebuild_cleaned)
    print(json.dumps(result, indent=2))
