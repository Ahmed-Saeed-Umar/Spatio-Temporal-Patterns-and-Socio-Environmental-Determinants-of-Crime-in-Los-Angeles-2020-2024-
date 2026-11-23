"""Load, clean and feature-engineer the LAPD crime dataset.

Run from the project root:
    python src/cleaning.py

Produces
    data/processed/crime_clean.parquet   cleaned + engineered table (fast to reload)
    outputs/tables/cleaning_log.json     every cleaning decision with row counts
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.api.types import union_categoricals

sys.path.insert(0, str(Path(__file__).resolve().parent))
from crime_groups import CRIME_CLASS, build_group_map  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = ROOT / "data" / "lapd_crime_2020_present.csv"
CLEAN_PATH = ROOT / "data" / "processed" / "crime_clean.parquet"
LOG_PATH = ROOT / "outputs" / "tables" / "cleaning_log.json"

# Cases from 2024 are excluded from analysis: the arrest rate collapses from ~9% to 2.5%
# during 2024 (recent cases are not yet resolved / still being recorded), and monthly
# volumes fall by more than half. Keeping them would bias every outcome statistic.
ANALYSIS_END = pd.Timestamp("2023-12-31")

MONTHS = {m: i + 1 for i, m in enumerate("Jan Feb Mar Apr May Jun Jul Aug Sep Oct Nov Dec".split())}

# LAPD victim-descent codes (documented in the LA open-data data dictionary)
DESCENT_GROUP = {
    "H": "Hispanic/Latin", "W": "White", "B": "Black",
    "A": "Asian", "C": "Asian", "D": "Asian", "F": "Asian", "J": "Asian",
    "K": "Asian", "L": "Asian", "V": "Asian", "Z": "Asian",
    "I": "Other", "G": "Other", "P": "Other", "S": "Other", "U": "Other", "O": "Other",
    "X": "Unknown",
}

# Hundreds digit of Weapon Used Cd (verified against Weapon Desc in the data)
WEAPON_GROUP = {
    "1": "Firearm", "2": "Knife / Cutting", "3": "Other Object",
    "4": "Physical Force", "5": "Unknown / Verbal / Other",
}

STATUS_GROUP = {
    "AA": "Arrest", "JA": "Arrest",
    "AO": "Cleared - No Arrest", "JO": "Cleared - No Arrest",
    "IC": "Investigation Continuing",
}

LA_LAT = (33.6, 34.9)     # sanity bounds for the City of Los Angeles
LA_LON = (-118.95, -117.6)


def parse_lapd_date(s: pd.Series) -> pd.Series:
    """Parse strings like '2021 Apr 11 12:00:00 AM' without depending on the OS locale.

    If the column is categorical (as in `load_raw`) only the ~1,800 distinct dates are parsed.
    """
    if isinstance(s.dtype, pd.CategoricalDtype):
        cats = pd.Series(s.cat.categories.astype(str))
        parsed = parse_lapd_date(cats)
        return s.map(dict(zip(cats, parsed))).astype("datetime64[ns]")
    s = s.astype(str)
    month = s.str.slice(5, 8).map(MONTHS).astype("Int64").astype(str).str.zfill(2)
    return pd.to_datetime(s.str.slice(0, 4) + "-" + month + "-" + s.str.slice(9, 11),
                          format="%Y-%m-%d", errors="coerce")


# Columns the project uses. Low-cardinality text is stored as 'category' so the 1M-row file fits
# comfortably in memory (about 0.4 GB instead of 3 GB as plain text). The file is read in chunks
# AS TEXT first - letting pandas infer categories would turn the time "0845" into the number 845.
_CAT = ["Date Rptd", "DATE OCC", "TIME OCC", "AREA NAME", "Rpt Dist No", "Part 1-2", "Crm Cd Desc",
        "Vict Sex", "Vict Descent", "Premis Desc", "Weapon Used Cd", "Status",
        "Crm Cd 1", "Crm Cd 2", "Crm Cd 3", "Crm Cd 4"]
_NUM = ["Vict Age", "LAT", "LON"]
_USECOLS = ["DR_NO", "Mocodes"] + _NUM + _CAT


def load_raw(path: Path = RAW_PATH, chunksize: int = 200_000) -> pd.DataFrame:
    """Read the raw CSV (only the columns used here) in a memory-friendly way."""
    parts = []
    for ch in pd.read_csv(path, usecols=_USECOLS, dtype=str, chunksize=chunksize):
        ch["DR_NO"] = ch["DR_NO"].astype("int64")
        for c in _NUM:
            ch[c] = pd.to_numeric(ch[c], errors="coerce")
        for c in _CAT:
            ch[c] = ch[c].astype("category")
        parts.append(ch)
    out = {}
    for c in _USECOLS:
        if c in _CAT:
            out[c] = union_categoricals([p[c] for p in parts])
        else:
            out[c] = pd.concat([p[c] for p in parts], ignore_index=True).to_numpy()
    return pd.DataFrame(out)


def profile_raw(path: Path = RAW_PATH, chunksize: int = 200_000) -> dict:
    """Quality profile of the untouched CSV, computed in chunks so memory stays small."""
    nulls, n, head = None, 0, None
    c = dict(age_le0=0, age_gt100=0, sex_not_mf=0, coords_00=0, mo_missing=0, weapon_missing=0)
    ids, month, quarter, year = [], None, None, None
    for ch in pd.read_csv(path, dtype=str, chunksize=chunksize):
        head = ch.head(3) if head is None else head
        nulls = ch.isna().sum() if nulls is None else nulls + ch.isna().sum()
        n += len(ch)
        age = pd.to_numeric(ch["Vict Age"], errors="coerce")
        c["age_le0"] += int((age <= 0).sum()); c["age_gt100"] += int((age > 100).sum())
        c["sex_not_mf"] += int((~ch["Vict Sex"].isin(["M", "F"])).sum())
        c["coords_00"] += int((pd.to_numeric(ch["LAT"], errors="coerce") == 0).sum())
        c["mo_missing"] += int(ch["Mocodes"].isna().sum())
        c["weapon_missing"] += int(ch["Weapon Used Cd"].isna().sum())
        ids.append(pd.to_numeric(ch["DR_NO"]).to_numpy())
        occ = parse_lapd_date(ch["DATE OCC"])
        arr = ch["Status"].isin(["AA", "JA"])
        m = occ.dt.to_period("M").value_counts()
        month = m if month is None else month.add(m, fill_value=0)
        for name, per in (("q", "Q"), ("y", "Y")):
            t = pd.DataFrame({"p": occ.dt.to_period(per), "a": arr}).groupby("p")["a"].agg(["sum", "size"])
            if name == "q":
                quarter = t if quarter is None else quarter.add(t, fill_value=0)
            else:
                year = t if year is None else year.add(t, fill_value=0)
    issues = [
        ("Victim age is 0 or negative (not a real age)", c["age_le0"]),
        ("Victim age above 100", c["age_gt100"]),
        ("Victim sex is not M or F (blank, X, H, '-')", c["sex_not_mf"]),
        ("Coordinates recorded as 0,0", c["coords_00"]),
        ("No Mocodes recorded", c["mo_missing"]),
        ("No weapon recorded", c["weapon_missing"]),
        ("Duplicate report numbers", int(n - len(np.unique(np.concatenate(ids))))),
    ]
    return {"rows": n, "cols": int(len(nulls)), "head": head, "null_pct": nulls / n * 100,
            "issues": issues, "monthly": month.sort_index().astype(int),
            "quarterly_arrest_pct": (quarter["sum"] / quarter["size"] * 100).sort_index(),
            "yearly": pd.DataFrame({"reports": year["size"].astype(int),
                                    "arrest_rate_%": year["sum"] / year["size"] * 100}).sort_index()}


def clean(raw: pd.DataFrame):
    """Return (clean_df, log). `log` records what each step changed."""
    log = {"rows_raw": int(len(raw)), "steps": []}

    def note(step, detail):
        log["steps"].append({"step": step, **detail})

    df = raw

    # 1. Duplicates -----------------------------------------------------------
    n = len(df)
    df = df.drop_duplicates(subset="DR_NO")
    note("Drop duplicate report numbers (DR_NO)", {"rows_removed": int(n - len(df))})

    # 2. Dates and time -------------------------------------------------------
    df["date_occ"] = parse_lapd_date(df["DATE OCC"])
    df["date_rptd"] = parse_lapd_date(df["Date Rptd"])
    n_bad = int(df["date_occ"].isna().sum() + df["date_rptd"].isna().sum())
    df = df.dropna(subset=["date_occ", "date_rptd"])
    note("Parse dates (locale-independent)", {"unparseable_rows_dropped": n_bad})

    df["hour"] = pd.to_numeric(df["TIME OCC"].astype(str).str.slice(0, 2), errors="coerce")
    df["year"] = df["date_occ"].dt.year
    df["month"] = df["date_occ"].dt.month
    df["dow"] = df["date_occ"].dt.day_of_week        # Monday = 0
    df["is_weekend"] = (df["dow"] >= 5).astype(int)
    df["part_of_day"] = pd.cut(df["hour"], [-1, 5, 11, 17, 23],
                               labels=["Night (0-5)", "Morning (6-11)",
                                       "Afternoon (12-17)", "Evening (18-23)"])

    lag = (df["date_rptd"] - df["date_occ"]).dt.days
    note("Report lag = date reported - date occurred",
         {"negative_lags_set_missing": int((lag < 0).sum())})
    df["report_lag_days"] = lag.where(lag >= 0)
    df["log_report_lag"] = np.log1p(df["report_lag_days"])

    # 3. Victim demographics --------------------------------------------------
    age = pd.to_numeric(df["Vict Age"], errors="coerce")
    bad_age = (age <= 0) | (age > 100)
    note("Victim age: 0, negative or >100 is not a real age -> missing",
         {"values_set_missing": int(bad_age.sum()), "share": round(float(bad_age.mean()), 4)})
    df["vict_age"] = age.where(~bad_age)
    df["age_group"] = pd.cut(df["vict_age"], [0, 17, 24, 34, 44, 54, 64, 100],
                             labels=["<18", "18-24", "25-34", "35-44", "45-54", "55-64", "65+"])

    sex = df["Vict Sex"].astype(object)
    note("Victim sex: only M/F kept; X, H, '-' and blanks -> 'Unknown'",
         {"set_unknown": int((~sex.isin(["M", "F"])).sum())})
    df["vict_sex"] = sex.where(sex.isin(["M", "F"]), "Unknown")

    df["vict_descent"] = df["Vict Descent"].astype(object).map(DESCENT_GROUP).fillna("Unknown")

    # 4. Location -------------------------------------------------------------
    lat = pd.to_numeric(df["LAT"], errors="coerce")
    lon = pd.to_numeric(df["LON"], errors="coerce")
    bad_geo = ~(lat.between(*LA_LAT) & lon.between(*LA_LON))
    note("Coordinates: 0,0 or outside Los Angeles -> missing",
         {"values_set_missing": int(bad_geo.sum())})
    df["lat"] = lat.where(~bad_geo)
    df["lon"] = lon.where(~bad_geo)
    df["area"] = df["AREA NAME"].astype(str).str.strip()

    # 5. Crime type -----------------------------------------------------------
    df["crime_desc"] = df["Crm Cd Desc"].astype(str).str.strip()
    gmap = build_group_map(df["crime_desc"])
    df["crime_group"] = df["crime_desc"].map(gmap)
    df["crime_class"] = df["crime_group"].map(CRIME_CLASS)
    df["part_1_2"] = pd.to_numeric(df["Part 1-2"].astype(str), errors="coerce").astype("Int64")
    extra = df[["Crm Cd 1", "Crm Cd 2", "Crm Cd 3", "Crm Cd 4"]].notna().sum(axis=1)
    df["n_crime_codes"] = extra
    note("Crime descriptions grouped by keyword rules",
         {"unique_descriptions": int(len(gmap)), "groups": int(df["crime_group"].nunique())})

    # 6. Weapon ---------------------------------------------------------------
    wcd = df["Weapon Used Cd"].astype(object)
    df["weapon_used"] = wcd.notna().astype(int)
    df["weapon_group"] = wcd.str.slice(0, 1).map(WEAPON_GROUP).fillna("None recorded")
    note("Weapon: blank Weapon Used Cd means none recorded (67% of rows)",
         {"rows_without_weapon": int(wcd.isna().sum())})

    # 7. Premise --------------------------------------------------------------
    prem = df["Premis Desc"].astype(object).str.strip().fillna("UNKNOWN")
    top = prem.value_counts().head(20).index
    df["premise"] = prem.where(prem.isin(top), "OTHER")
    note("Premise: keep the 20 most common, rest -> OTHER", {"kept": 20})

    # 8. Modus operandi codes -------------------------------------------------
    df["mocodes"] = df["Mocodes"].fillna("")
    df["n_mo"] = df["mocodes"].str.split().str.len().fillna(0).astype(int)
    note("Mocodes kept as text for the MO-pattern analysis",
         {"rows_without_mocodes": int((df["n_mo"] == 0).sum())})

    # 9. Target: case outcome -------------------------------------------------
    df["outcome"] = df["Status"].astype(object).map(STATUS_GROUP)
    n_unk = int(df["outcome"].isna().sum())
    df = df.dropna(subset=["outcome"])
    note("Outcome from Status (AA/JA = Arrest); rare UNK/CC statuses dropped",
         {"rows_removed": n_unk})
    df["arrest"] = (df["outcome"] == "Arrest").astype(int)

    # 10. Analysis window -----------------------------------------------------
    n = len(df)
    df = df[df["date_occ"] <= ANALYSIS_END]
    note("Restrict to incidents up to 2023-12-31 (2024 is incomplete)",
         {"rows_removed": int(n - len(df))})

    keep = ["DR_NO", "date_occ", "date_rptd", "year", "month", "dow", "is_weekend", "hour",
            "part_of_day", "area", "Rpt Dist No", "crime_desc", "crime_group", "crime_class",
            "part_1_2", "n_crime_codes", "vict_age", "age_group", "vict_sex", "vict_descent",
            "premise", "weapon_used", "weapon_group", "mocodes", "n_mo", "report_lag_days",
            "log_report_lag", "lat", "lon", "outcome", "arrest"]
    df = df[keep].rename(columns={"DR_NO": "dr_no", "Rpt Dist No": "rpt_dist"})
    for c in ["area", "crime_desc", "crime_group", "crime_class", "vict_sex", "vict_descent",
              "premise", "weapon_group", "outcome"]:
        df[c] = df[c].astype("category")
    df = df.reset_index(drop=True)

    log["rows_clean"] = int(len(df))
    log["arrest_rate"] = round(float(df["arrest"].mean()), 4)
    log["date_range"] = [str(df["date_occ"].min().date()), str(df["date_occ"].max().date())]
    return df, log


def build(force: bool = False) -> pd.DataFrame:
    """Return the clean table, creating it from the raw CSV the first time."""
    if CLEAN_PATH.exists() and not force:
        return pd.read_parquet(CLEAN_PATH)
    if not RAW_PATH.exists():
        raise FileNotFoundError(f"Dataset not found at {RAW_PATH}. See README.md, section 'Data'.")
    print("Reading raw CSV (about 1 million rows) ...")
    df, log = clean(load_raw())
    CLEAN_PATH.parent.mkdir(parents=True, exist_ok=True)
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(CLEAN_PATH, index=False)
    LOG_PATH.write_text(json.dumps(log, indent=2))
    print(f"Saved {len(df):,} clean rows -> {CLEAN_PATH}")
    return df


if __name__ == "__main__":
    build(force=True)
