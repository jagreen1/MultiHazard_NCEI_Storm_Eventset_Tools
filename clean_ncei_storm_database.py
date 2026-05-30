"""
NCEI Storm Multihazard Eventset Data Cleaning Pipeline

Created March 21st, 2025
@author: Joshua Green - University of Southampton

Please cite this script/dataset if used in any publications:
Green, J. (2025) NCEI Storm Multihazard Eventset.

This module processes and standardizes storm event data from NOAA NCEI's Storm Events Database,
applying data quality checks, standardization rules, timezone conversions, and inflation adjustments.

Input data can be downloaded from:
- https://www.ncdc.noaa.gov/stormevents/ftp.jsp
- https://www.ncei.noaa.gov/pub/data/swdi/stormevents/csvfiles/
- ftp://ftp.ncei.noaa.gov/pub/data/swdi/stormevents/csvfiles/
"""

from datetime import datetime
import glob
import os
import numpy as np
import pandas as pd
import pytz

pd.set_option('display.max_columns', None)


# ============================================================================
# TIMEZONE AND MAPPING CONFIGURATIONS
# ============================================================================

TIMEZONE_SUBSTITUTIONS = {
    "CDT": "CST",
    "CSC": "CST",
    "EDT": "EST",
    "GMT": "CST",
    "GST": "CHST",
    "MDT": "MST",
    "PDT": "PST",
    "SCT": "CST",
}

STATE_TIMEZONE_MAP = {
    "GUAM": "CHST",
    "ALASKA": "AKST",
    "ATLANTIC": "EST",
    "HAWAII": "HST",
    "PUERTO RICO": "AST",
    "VIRGIN ISLANDS": "AST",
}

UNKNOWN_TIMEZONE_FALLBACK = {
    "HAWAII": "HST",
    "OKLAHOMA": "CST",
    "MASSACHUSETTS": "EST",
    "GEORGIA": "EST",
    "ILLINOIS": "CST",
}

ABBREV_TO_IANA = {
    'EST': 'America/New_York',
    'CST': 'America/Chicago',
    'MST': 'America/Denver',
    'PST': 'America/Los_Angeles',
    'HST': 'Pacific/Honolulu',
    'CHST': 'Pacific/Guam',
    'SST': 'Pacific/Samoa',
    'AKST': 'America/Anchorage',
    'AST': 'America/Puerto_Rico',
}

SOURCE_ACRONYMS = ["Asos", "Awos", "Awss", "Nws", "C-Man", "Raws", "Shave", "Snotel", "Wlon"]

SOURCE_STANDARDIZATION = {
    "Arpt Equip(AWOS,ASOS)": "AWOS,ASOS,Mesonet,Etc",
    "Coastal Observing Station": "Coast Guard",
    "Cocorahs": "CoCoRaHS",
    "Coop Observer": "Cooperative Network Observer",
    "Coop Station": "Cooperative Network Observer",
    "Dept Of Highways": "Department Of Highways",
    "Fire Dept/Rescue Squad": "Fire Department/Rescue",
    "General Public": "Public",
    "Govt Official": "State Official",
    "Manual Input": "Unknown",
    "Meteorologist(Non NWS)": "Public",
    "NWS Employee(Off Duty)": "NWS Employee",
    "Npop": "Unknown",
    "Official NWS Obs.": "Official NWS Observations",
}

EVENT_TYPE_STANDARDIZATION = {
    r"^HAIL.*": "Hail",
    r"^High Snow$": "Heavy Snow",
    r"^Hurricane$": "Hurricane",
    r"^OTHER$": "Dust Devil",
    r"^THUNDERSTORM WIND.*": "Thunderstorm Wind",
    r"^TORNADO.*": "Tornado",
    r"^Volcanic Ashfall.*": "Volcanic Ash",
    "Hurricane (Typhoon)": "Hurricane/Typhoon",
    "TropicalDepression": "Tropical Depression"
}

HAZARD_ACRONYM_MAP = {
    "Avalanche": "av",
    "Blizzard": "bz",
    "Coastal Flood": "cfl",
    "Lakeshore Flood": "cfl",
    "Cold/Wind Chill": "cw",
    "Extreme Cold/Wind Chill": "cw",
    "Dust Devil": "tn",
    "Debris Flow": "df",
    "Drought": "dr",
    "Dust Storm": "ds",
    "High Wind": "ew",
    "Strong Wind": "ew",
    "Heavy Wind": "ew",
    "Funnel Cloud": "fc",
    "Frost/Freeze": "ff",
    "Freezing Fog": "fg",
    "Dense Fog": "fg",
    "Flood": "fl",
    "Hail": "hl",
    "High Surf": "hs",
    "Hurricane/Typhoon": "tc",
    "Hurricane": "tc",
    "Heat": "hw",
    "Excessive Heat": "hw",
    "Ice Storm": "is",
    "Lake-Effect Snow": "sn",
    "Astronomical Low Tide": "lt",
    "Lightning": "ltn",
    "Marine High Wind": "mew",
    "Marine Strong Wind": "mew",
    "Marine Dense Fog": "mfg",
    "Marine Hail": "mhl",
    "Marine Hurricane/Typhoon": "mtc",
    "Marine Lightning": "mltn",
    "Marine Tropical Storm": "mtc",
    "Marine Tropical Depression": "mtc",
    "Marine Thunderstorm Wind": "mtw",
    "Northern Lights": "nl",
    "Heavy Rain": "rn",
    "Flash Flood": "pfl",
    "Rip Current": "rc",
    "Seiche": "se",
    "Sleet": "sl",
    "Dense Smoke": "sm",
    "Heavy Snow": "sn",
    "Storm Surge/Tide": "sst",
    "Sneakerwave": "swv",
    "Tropical Storm": "tc",
    "Tropical Depression": "tc",
    "Tornado": "tn",
    "Tsunami": "ts",
    "Thunderstorm Wind": "tw",
    "Volcanic Ash": "vo",
    "Wildfire": "wf",
    "Waterspout": "wp",
    "Winter Storm": "ws",
    "Winter Weather": "ww",
}

HAZARD_NAME_MAP = {
    "av": "Avalanche",
    "bz": "Blizzard",
    "cfl": "Coastal Flood",
    "cw": "Cold/Wind Chill",
    "df": "Debris Flow",
    "dr": "Drought",
    "ds": "Dust Storm",
    "ew": "Extreme Wind",
    "fc": "Funnel Cloud",
    "ff": "Frost/Freeze",
    "fg": "Fog",
    "fl": "Flood",
    "hl": "Hail",
    "hs": "High Surf",
    "hw": "Heat",
    "is": "Ice Storm",
    "lt": "Astronomical Low Tide",
    "ltn": "Lightning",
    "mew": "Marine Extreme Wind",
    "mfg": "Marine Fog",
    "mhl": "Marine Hail",
    "mtc": "Marine Tropical Cyclone",
    "mltn": "Marine Lightning",
    "mtw": "Marine Thunderstorm Wind",
    "nl": "Northern Lights",
    "rn": "Rain",
    "pfl": "Flash Flood",
    "rc": "Rip Current",
    "se": "Seiche",
    "sl": "Sleet",
    "sm": "Smoke",
    "sn": "Snow",
    "sst": "Storm Surge/Tide",
    "swv": "Sneakerwave",
    "tc": "Tropical Cyclone",
    "tn": "Tornado",
    "ts": "Tsunami",
    "tw": "Thunderstorm Wind",
    "vo": "Volcanic Ash",
    "wf": "Wildfire",
    "wp": "Waterspout",
    "ws": "Winter Storm",
    "ww": "Winter Weather",
}

OUTPUT_COLUMNS = [
    "EPISODE_ID", "EVENT_ID", "GEOID", "STATE", "STATE_FIPS", "EVENT_TYPE",
    "ORIG_EVENT_TYPE", "HAZARD", "CZ_TYPE", "CZ_FIPS", "CZ_NAME",
    "BEGIN_DATETIME_UTC", "END_DATETIME_UTC", "BEGIN_DATETIME", "END_DATETIME",
    "start_year", "end_year", "WFO", "CZ_TIMEZONE", "INJURIES_DIRECT",
    "INJURIES_INDIRECT", "DEATHS_DIRECT", "DEATHS_INDIRECT", "DAMAGE_PROPERTY",
    "DAMAGE_CROPS", "ADJ_DAMAGE_PROPERTY", "ADJ_DAMAGE_CROPS", "TOTAL_INJURIES",
    "TOTAL_DEATHS", "TOTAL_ADJ_DAMAGE", "SOURCE", "MAGNITUDE", "MAGNITUDE_TYPE",
    "FLOOD_CAUSE", "CATEGORY", "TOR_F_SCALE", "TOR_LENGTH", "TOR_WIDTH",
    "TOR_OTHER_WFO", "TOR_OTHER_CZ_STATE", "TOR_OTHER_CZ_FIPS", "TOR_OTHER_CZ_NAME",
    "BEGIN_RANGE", "BEGIN_AZIMUTH", "BEGIN_LOCATION", "END_RANGE", "END_AZIMUTH",
    "END_LOCATION", "BEGIN_LAT", "BEGIN_LON", "END_LAT", "END_LON", "DATA_SOURCE",
    "EPISODE_NARRATIVE", "EVENT_NARRATIVE",
]


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def load_storm_data(input_dir, pattern="*details*.csv.gz"):
    """Load and concatenate compressed CSV files from input directory."""
    csv_files = glob.glob(os.path.join(input_dir, pattern))
    if not csv_files:
        raise FileNotFoundError(f"No files matching pattern '{pattern}' found in {input_dir}")
    dataframes = [pd.read_csv(f, low_memory=False, compression='gzip') for f in csv_files]
    return pd.concat(dataframes, ignore_index=True)


def load_remote_data(url, delimiter=','):
    """Load reference data from remote GitHub repository."""
    try:
        return pd.read_csv(url, delimiter=delimiter)
    except Exception as e:
        raise ConnectionError(f"Failed to load data from {url}: {str(e)}")


def filter_required_fields(df):
    """Remove records with missing critical fields."""
    initial_count = len(df)
    
    df = df[df["EVENT_ID"].notna()]
    df["EVENT_ID"] = df["EVENT_ID"].astype(int)
    df = df[df["STATE"].notna()]
    df = df[df["STATE_FIPS"].notna()]
    df["STATE_FIPS"] = df["STATE_FIPS"].astype(int)
    df = df[df["EVENT_TYPE"].notna()]
    df = df[df["CZ_FIPS"].notna()]
    
    removed = initial_count - len(df)
    print(f"Filtered required fields: removed {removed} records")
    return df


def standardize_location_fields(df):
    """Clean and standardize county zone identifiers."""
    df['CZ_NAME'] = df['CZ_NAME'].astype(str).str.strip().str.upper()
    df["CZ_FIPS"] = df["CZ_FIPS"].astype(str).str.strip()
    
    # Convert NWS zones (CZ_TYPE='Z') to county FIPS codes
    cz_type_c = df[df["CZ_TYPE"] == "C"]
    mapping = cz_type_c.set_index(["STATE_FIPS", "STATE", "CZ_NAME"])["CZ_FIPS"].to_dict()
    
    def map_cz_fips(row):
        if row["CZ_TYPE"] == "Z":
            key = (row["STATE_FIPS"], row["STATE"], row["CZ_NAME"])
            return mapping.get(key, row["CZ_FIPS"])
        return row["CZ_FIPS"]
    
    df["CZ_FIPS"] = df.apply(map_cz_fips, axis=1)
    df["CZ_FIPS"] = df["CZ_FIPS"].astype(str).str.zfill(3)
    df["STATE_FIPS"] = df["STATE_FIPS"].astype(str).str.zfill(2)
    df["GEOID"] = df["STATE_FIPS"] + df["CZ_FIPS"]
    
    return df


def standardize_source_field(df):
    """Standardize event source reporting names and acronyms."""
    df.SOURCE = df.SOURCE.str.title()
    
    for acronym in SOURCE_ACRONYMS:
        pattern = f"\\b{acronym}\\b"
        replacement = acronym.upper()
        df.SOURCE = df.SOURCE.str.strip().replace(pattern, replacement, regex=True)
    
    for original, replacement in SOURCE_STANDARDIZATION.items():
        df.SOURCE = df.SOURCE.str.replace(original, replacement, regex=False)
    
    return df


def standardize_event_types(df):
    """Standardize and abbreviate hazard event type names."""
    for original, replacement in EVENT_TYPE_STANDARDIZATION.items():
        df.EVENT_TYPE = df.EVENT_TYPE.str.strip().replace(original, replacement, regex=True)
    
    df["ORIG_EVENT_TYPE"] = df["EVENT_TYPE"].copy()
    df["HAZARD"] = df["EVENT_TYPE"].map(HAZARD_ACRONYM_MAP)
    df["EVENT_TYPE"] = df["HAZARD"].map(HAZARD_NAME_MAP)
    
    return df


def standardize_timezones(df):
    """Standardize timezone abbreviations and assign state-based defaults."""
    def assign_state_tz(state):
        s = (state or "").strip().upper()
        for key, tz in STATE_TIMEZONE_MAP.items():
            if key in s:
                return tz
        return None
    
    picks = df["STATE"].apply(assign_state_tz)
    df.loc[picks.notna(), "CZ_TIMEZONE"] = picks[picks.notna()]
    
    df.CZ_TIMEZONE = df.CZ_TIMEZONE.str.strip().replace(r"-*\d*$", "", regex=True).str.upper()
    
    for original, replacement in TIMEZONE_SUBSTITUTIONS.items():
        df.CZ_TIMEZONE = df.CZ_TIMEZONE.str.replace(original, replacement, regex=False)
    
    for idx, row in df.query('CZ_TIMEZONE=="UNK"').iterrows():
        df.at[idx, "CZ_TIMEZONE"] = UNKNOWN_TIMEZONE_FALLBACK.get(row.STATE, "UNK")
    
    return df


def create_datetime_from_components(df, prefix):
    """Reconstruct datetime from YEARMONTH, DAY, and TIME fields."""
    dt_components = pd.to_datetime({
        "year": df[f"{prefix}YEARMONTH"] // 100,
        "month": df[f"{prefix}YEARMONTH"] % 100,
        "day": df[f"{prefix}DAY"],
        "hour": df[f"{prefix}TIME"] // 100,
        "minute": df[f"{prefix}TIME"] % 100,
    })
    return pd.to_datetime(dt_components)


def convert_to_utc(df, dt_col, tz_col):
    """Convert local datetime to UTC using timezone abbreviation."""
    def row_to_utc(row):
        dt = row[dt_col]
        tzabbr = row.get(tz_col)
        
        if pd.isna(dt) or pd.isna(tzabbr):
            return pd.NaT
        
        iana = ABBREV_TO_IANA.get(tzabbr)
        if iana is None:
            return pd.NaT
        
        ts = pd.to_datetime(dt, errors="coerce")
        if pd.isna(ts):
            return pd.NaT
        
        if ts.tzinfo is not None and ts.tzinfo.utcoffset(ts) is not None:
            return ts.tz_convert("UTC")
        
        try:
            return ts.tz_localize(iana, nonexistent="shift_forward", ambiguous=False).tz_convert("UTC")
        except Exception as e:
            ename = type(e).__name__
            if ename in ("NonExistentTimeError", "AmbiguousTimeError"):
                return ts.tz_localize(iana, ambiguous=False, nonexistent="shift_forward").tz_convert("UTC")
            return pd.NaT
    
    return df.apply(row_to_utc, axis=1)


def parse_damage_value(column):
    """Parse and scale damage values (e.g., '100K', '2.5M') to numeric."""
    price = column[column.notna()].astype(str).str.upper()
    valid_price = r"^[\d.]+[KMB]?$"
    price = price[price.str.contains(valid_price, regex=True)]
    
    has_K = price.str.contains("K")
    has_M = price.str.contains("M")
    has_B = price.str.contains("B")
    
    price = price.str.replace(r"[KMB]", "", regex=True).astype(float)
    scale = np.select([has_K, has_M, has_B], [1000, 1_000_000, 1_000_000_000], 1)
    
    return scale * price


def clean_impact_values(df):
    """Standardize impact fields (injuries, deaths, damage) and remove invalid values."""
    impact_cols = ['DEATHS_DIRECT', 'DEATHS_INDIRECT', 'INJURIES_DIRECT', 'INJURIES_INDIRECT']
    damage_cols = ['DAMAGE_CROPS', 'DAMAGE_PROPERTY']
    
    for col in impact_cols + damage_cols:
        df[col] = df[col].fillna(0).astype(int)
        df.loc[df[col] < 0, col] = 0
    
    return df


def apply_inflation_adjustment(df, cpi_lookup, target_year):
    """Adjust historical damage values to target year dollars using CPI."""
    df['INFLATION_YEAR'] = df['start_year']
    inflation_cpi_target = cpi_lookup.get(target_year)
    
    if inflation_cpi_target is None:
        raise ValueError(f"Target year {target_year} not found in CPI lookup")
    
    df['ADJ_DAMAGE_PROPERTY'] = df.apply(
        lambda row: round(row['DAMAGE_PROPERTY'] * (inflation_cpi_target / cpi_lookup[row['INFLATION_YEAR']])),
        axis=1
    )
    df['ADJ_DAMAGE_CROPS'] = df.apply(
        lambda row: round(row['DAMAGE_CROPS'] * (inflation_cpi_target / cpi_lookup[row['INFLATION_YEAR']])),
        axis=1
    )
    
    return df


def compute_aggregate_impacts(df):
    """Calculate total impacts across direct/indirect and property/crop categories."""
    df['TOTAL_ADJ_DAMAGE'] = (df['ADJ_DAMAGE_PROPERTY'] + df['ADJ_DAMAGE_CROPS']).fillna(0)
    df['TOTAL_DEATHS'] = (df['DEATHS_DIRECT'] + df['DEATHS_INDIRECT']).fillna(0)
    df['TOTAL_INJURIES'] = (df['INJURIES_DIRECT'] + df['INJURIES_INDIRECT']).fillna(0)
    
    return df


def compute_location_completeness(df):
    """Calculate data completeness metrics for location information."""
    total_events = len(df)
    
    metrics = {
        'with_coordinates': (df.BEGIN_LAT.notna() & df.BEGIN_LON.notna()).sum(),
        'with_cz_name': df.CZ_NAME.notna().sum(),
        'with_cz_fips': df.CZ_FIPS.notna().sum(),
        'with_state_fips': df.STATE_FIPS.notna().sum(),
        'with_state_name': df.STATE.notna().sum(),
        'with_all_location': (
            df.CZ_NAME.notna() & df.CZ_FIPS.notna() & df.STATE_FIPS.notna() &
            df.STATE.notna() & df.BEGIN_LAT.notna() & df.BEGIN_LON.notna()
        ).sum(),
    }
    
    print("\n=== Data Quality Report ===")
    print(f"Total events: {total_events}")
    for key, count in metrics.items():
        pct = (count / total_events * 100) if total_events > 0 else 0
        print(f"{key}: {count} ({pct:.1f}%)")
    
    return metrics


# ============================================================================
# MAIN PROCESSING FUNCTION
# ============================================================================

def clean_ncei_storm_database(input_dir, output_dir, inflation_target_year=2024):
    """
    Process and clean NCEI Storm Events Database.
    
    Parameters
    ----------
    input_dir : str
        Directory containing raw compressed storm event CSV files
    output_dir : str
        Directory for saving cleaned parquet output files
    inflation_target_year : int
        Year to normalize damage values to (default: 2024)
    """
    
    print(f"Starting NCEI Storm Database cleaning pipeline at {datetime.now()}")
    
    # Create output directory if needed
    os.makedirs(output_dir, exist_ok=True)
    
    # Load remote reference data
    print("Loading reference data...")
    nws_to_fips_df = load_remote_data(
        'https://raw.githubusercontent.com/jagreen1/NCEI_Storm_Multihazard_Eventset/'
        'refs/heads/main/NWS_Zone_to_County_FIPS_bp18mr25.dbx.txt.csv'
    )
    cpi_df = load_remote_data(
        'https://raw.githubusercontent.com/jagreen1/NCEI_Storm_Multihazard_Eventset/'
        'refs/heads/main/US_BLS_CPI_Inflation_1950-2024.txt',
        delimiter='\t'
    )
    cpi_lookup = cpi_df.set_index('Year')['Annual'].to_dict()
    
    # Load and process main dataset
    print("Loading storm event data...")
    df = load_storm_data(input_dir)
    print(f"Loaded {len(df)} initial records")
    
    # Data cleaning pipeline
    print("Filtering required fields...")
    df = filter_required_fields(df)
    
    print("Standardizing location fields...")
    df = standardize_location_fields(df)
    df = df.drop_duplicates()
    
    print("Standardizing source field...")
    df = standardize_source_field(df)
    
    print("Standardizing event types...")
    df = standardize_event_types(df)
    
    print("Standardizing timezones...")
    df = standardize_timezones(df)
    
    # Filter events with unknown timezone
    unknown_tz = df[df["CZ_TIMEZONE"] == "UNK"].copy()
    df = df[df["CZ_TIMEZONE"] != "UNK"]
    print(f"Excluded {len(unknown_tz)} events with unknown timezone")
    
    # Create and convert datetime fields
    print("Processing datetime fields...")
    legacy_cols = [
        "BEGIN_YEARMONTH", "BEGIN_DAY", "BEGIN_TIME", "BEGIN_DATE_TIME",
        "END_YEARMONTH", "END_DAY", "END_TIME", "END_DATE_TIME",
        "MONTH_NAME", "YEAR"
    ]
    
    df["BEGIN_DATETIME"] = create_datetime_from_components(df, "BEGIN_")
    df["END_DATETIME"] = create_datetime_from_components(df, "END_")
    df = df.drop(columns=legacy_cols, errors='ignore')
    
    df["start_year"] = df["BEGIN_DATETIME"].dt.year
    df["end_year"] = df["END_DATETIME"].dt.year
    
    df['BEGIN_DATETIME_UTC'] = convert_to_utc(df, 'BEGIN_DATETIME', 'CZ_TIMEZONE')
    df['END_DATETIME_UTC'] = convert_to_utc(df, 'END_DATETIME', 'CZ_TIMEZONE')
    
    # Process damage values
    print("Processing damage values...")
    df.DAMAGE_PROPERTY = parse_damage_value(df.DAMAGE_PROPERTY)
    df.DAMAGE_CROPS = parse_damage_value(df.DAMAGE_CROPS)
    
    print("Cleaning impact fields...")
    df = clean_impact_values(df)
    
    # Apply inflation adjustment
    print(f"Adjusting damages to {inflation_target_year} dollars...")
    df = apply_inflation_adjustment(df, cpi_lookup, inflation_target_year)
    
    # Compute aggregate fields
    df = compute_aggregate_impacts(df)
    
    # Compute and display data quality metrics
    compute_location_completeness(df)
    
    # Finalize dataset
    df = df.reindex(columns=OUTPUT_COLUMNS)
    df = df.sort_values(["BEGIN_DATETIME_UTC", "CZ_FIPS"], ascending=[True, True])
    df = df.drop_duplicates()
    
    # Save full dataset (1950-2024)
    print(f"\nSaving full dataset (1950-2024)...")
    output_full = os.path.join(output_dir, "NCEI_Storm_Database_Cleaned_Details_1950-2024_v2.parquet")
    df.to_parquet(output_full, compression="gzip")
    print(f"Saved to {output_full}")
    
    # Save subset (1996-2024)
    print("Saving subset dataset (1996-2024)...")
    df_subset = df[df["BEGIN_DATETIME_UTC"] >= pd.Timestamp("1996-01-01 00:00:00", tz="UTC")]
    df_subset = df_subset.sort_values(["BEGIN_DATETIME_UTC", "CZ_FIPS"], ascending=[True, True])
    df_subset = df_subset.drop_duplicates()
    
    output_subset = os.path.join(output_dir, "NCEI_Storm_Database_Cleaned_Details_1996-2024_v2.parquet")
    df_subset.to_parquet(output_subset, compression="gzip")
    print(f"Saved to {output_subset}")
    
    print(f"\nSuccessfully completed NCEI Storm Database cleaning at {datetime.now()}")
    return df, df_subset


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    # User-defined input/output directories
    INPUT_DIR = r"C:\path\to\input\directory"  # Replace with actual path
    OUTPUT_DIR = r"C:\path\to\output\directory"  # Replace with actual path
    INFLATION_YEAR = 2024
    
    df_full, df_1996 = clean_ncei_storm_database(INPUT_DIR, OUTPUT_DIR, INFLATION_YEAR)
