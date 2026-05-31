###########################################################################
"""
Published 2026
@author: Joshua Green - University of Southampton

Please cite the datasets and preprocessing script if used in any publications:
Green, J. (2026) NCEI Storm Multihazard Eventset. [Dataset]. Zenodo. https://doi.org/10.5281/zenodo.20285674.
Green, J. (2026) MultiHazard_NCEI_Storm_Eventset_Tools. Github. https://github.com/jagreen1/MultiHazard_NCEI_Storm_Eventset_Tools.

This script uses the cleaned NCEI Storm Database to create four output event sets:
- (AE) All Hazard Events
- (SH) Single-Hazard Only Events
- (MHP) Multi-Hazard Event Pairs
- (MH) Unique Multi-Hazard Events

User defined parameters allow for customization of event sets:
- eventset duration (start/end year)
- hazard type (list of hazard acronyms)
- multi-hazard time lag (temporal overlap period in days)
- minimum hazard impact filter thresholds (injury, death, building damage, crop damage)

List of possible hazard types and acronyms
"av": "Avalanche"
"bz": "Blizzard"
"cfl": "Coastal Flood"
"cw": "Cold/Wind Chill"
"df": "Debris Flow"
"dr": "Drought"
"ds": "Dust Storm"
"ew": "Extreme Wind"
"fc": "Funnel Cloud"
"ff": "Frost/Freeze"
"fg": "Fog"
"fl": "Flood"
"hl": "Hail"
"hs": "High Surf"
"hw": "Heat"
"is": "Ice Storm"
"lt": "Astronomical Low Tide"
"ltn": "Lightning"
"mew": "Marine Extreme Wind"
"mfg": "Marine Fog"
"mhl": "Marine Hail"
"mtc": "Marine Tropical Cyclone"
"mltn": "Marine Lightning"
"mtw": "Marine Thunderstorm Wind"
"nl": "Northern Lights"
"rn": "Rain"
"pfl": "Flash Flood"
"rc": "Rip Current"
"se": "Seiche"
"sl": "Sleet"
"sm": "Smoke"
"sn": "Snow"
"sst": "Storm Surge/Tide"
"swv": "Sneakerwave"
"tc": "Tropical Cyclone"
"tn": "Tornado"
"ts": "Tsunami"
"tw": "Thunderstorm Wind"
"vo": "Volcanic Ash"
"wf": "Wildfire"
"wp": "Waterspout"
"ws": "Winter Storm"
"ww": "Winter Weather"
"""
###########################################################################

import os
import pandas as pd
import geopandas as gpd
import numpy as np
from tqdm import tqdm
import pickle


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def create_folder_if_not_exists(folder_path):
    """Create folder if it does not exist."""
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"Folder created at: {folder_path}")
    else:
        print(f"Folder already exists at: {folder_path}")


def load_and_prepare_data(parquet_path):
    """Load cleaned NCEI storm database and prepare basic columns."""
    df = pd.read_parquet(parquet_path)
    
    # Format FIPS codes with leading zeros and create GEOID
    df["CZ_FIPS"] = df["CZ_FIPS"].astype(str).str.zfill(3)
    df["STATE_FIPS"] = df["STATE_FIPS"].astype(str).str.zfill(2)
    df["GEOID"] = df["STATE_FIPS"].astype(str).str.zfill(2) + df["CZ_FIPS"].astype(str).str.zfill(3)
    
    return df


def apply_qa_qc_checks(df):
    """Remove records with null critical fields."""
    critical_fields = [
        "EVENT_ID", "STATE", "STATE_FIPS", "EVENT_TYPE", "CZ_FIPS",
        "BEGIN_DATETIME_UTC", "END_DATETIME_UTC", "BEGIN_DATETIME", "END_DATETIME"
    ]
    
    for field in critical_fields:
        df = df[~df[field].isnull()]
    
    return df


def clean_impact_columns(df):
    """Clean and standardize impact columns (injuries, deaths, damage)."""
    impact_cols = [
        'DEATHS_DIRECT', 'DEATHS_INDIRECT', 'INJURIES_DIRECT', 'INJURIES_INDIRECT',
        'DAMAGE_CROPS', 'DAMAGE_PROPERTY', 'ADJ_DAMAGE_CROPS', 'ADJ_DAMAGE_PROPERTY',
        'TOTAL_INJURIES', 'TOTAL_DEATHS', 'TOTAL_ADJ_DAMAGE'
    ]
    
    for col in impact_cols:
        df[col] = df[col].fillna(0).astype(int) if col != 'TOTAL_ADJ_DAMAGE' else df[col].fillna(0)
        df.loc[df[col] < 0, col] = 0
    
    return df


def reindex_columns(df):
    """Reorder dataframe columns to standard format."""
    column_order = [
        "EPISODE_ID", "EVENT_ID", "GEOID", "STATE", "STATE_FIPS", "EVENT_TYPE", "HAZARD",
        "CZ_TYPE", "CZ_FIPS", "CZ_NAME", "BEGIN_DATETIME_UTC", "END_DATETIME_UTC",
        "BEGIN_DATETIME", "END_DATETIME", "start_year", "end_year", "WFO", "CZ_TIMEZONE",
        "INJURIES_DIRECT", "INJURIES_INDIRECT", "DEATHS_DIRECT", "DEATHS_INDIRECT",
        "DAMAGE_PROPERTY", "DAMAGE_CROPS", "ADJ_DAMAGE_PROPERTY", "ADJ_DAMAGE_CROPS",
        'TOTAL_INJURIES', 'TOTAL_DEATHS', 'TOTAL_ADJ_DAMAGE', "SOURCE", "MAGNITUDE",
        "MAGNITUDE_TYPE", "FLOOD_CAUSE", "CATEGORY", "TOR_F_SCALE", "TOR_LENGTH",
        "TOR_WIDTH", "TOR_OTHER_WFO", "TOR_OTHER_CZ_STATE", "TOR_OTHER_CZ_FIPS",
        "TOR_OTHER_CZ_NAME", "BEGIN_RANGE", "BEGIN_AZIMUTH", "BEGIN_LOCATION",
        "END_RANGE", "END_AZIMUTH", "END_LOCATION", "BEGIN_LAT", "BEGIN_LON",
        "END_LAT", "END_LON", "DATA_SOURCE", "EPISODE_NARRATIVE", "EVENT_NARRATIVE",
    ]
    return df.reindex(columns=column_order)


def filter_by_temporal_range(df, start_year, end_year):
    """Filter events by year range using UTC times."""
    return df[
        (df["BEGIN_DATETIME_UTC"].dt.year >= start_year) &
        (df["END_DATETIME_UTC"].dt.year <= end_year)
    ]


def filter_by_state_and_zone(df):
    """Remove unwanted states and marine zone-only events."""
    exclusion_states = [
        "ALASKA", "AMERICAN SAMOA", "ATLANTIC NORTH", "ATLANTIC SOUTH", "E PACIFIC",
        "GUAM WATERS", "GUAM", "GULF OF ALASKA", "GULF OF MEXICO", "HAWAII WATERS",
        "HAWAII", "LAKE ERIE", "LAKE HURON", "LAKE MICHIGAN", "LAKE ONTARIO",
        "LAKE ST CLAIR", "LAKE SUPERIOR", "PUERTO RICO", "ST LAWRENCE R", "VIRGIN ISLANDS",
    ]
    df = df[~df["STATE"].isin(exclusion_states)]
    df = df[df['CZ_TYPE'] != 'M']  # Remove marine zones
    return df


def filter_by_hazard_type(df, inclusion_filter):
    """Filter events by hazard type."""
    return df[df['HAZARD'].isin(inclusion_filter)]


def filter_by_impact(df, inj, dth, c, p):
    """Filter events by minimum impact thresholds."""
    return df[
        (df["INJURIES_DIRECT"] >= inj) |
        (df["INJURIES_INDIRECT"] >= inj) |
        (df["DEATHS_DIRECT"] >= dth) |
        (df["DEATHS_INDIRECT"] >= dth) |
        (df["ADJ_DAMAGE_CROPS"] >= c * 1000) |
        (df["ADJ_DAMAGE_PROPERTY"] >= p * 1000)
    ]


def datetime_ranges_overlap_with_lag(start1, end1, start2, end2, lag):
    """Check if two datetime ranges overlap with a time lag."""
    return max(start1 - lag, start2 - lag) <= min(end1 + lag, end2 + lag)


def unique_pairs(pairs):
    """Remove duplicate event pairs."""
    unique_set = set()
    unique_list = []
    for pair in pairs:
        sorted_pair = tuple(sorted(pair))
        if sorted_pair not in unique_set:
            unique_set.add(sorted_pair)
            unique_list.append(pair)
    return unique_list


def identify_overlapping_events(df, time_lag):
    """Identify overlapping hazard events within a county."""
    df["OVERLAPPING_EVENTS"] = [[] for _ in range(len(df))]
    overlapping_event_pairs = []
    
    for idx, row in df.iterrows():
        subset = df[
            (df["CZ_FIPS"] == row["CZ_FIPS"]) &
            (df["CZ_NAME"] == row["CZ_NAME"]) &
            (df["STATE_FIPS"] == row["STATE_FIPS"])
        ]
        
        for _, other in subset.iterrows():
            if row["EVENT_ID"] == other["EVENT_ID"]:
                continue
            
            if not datetime_ranges_overlap_with_lag(
                row["BEGIN_DATETIME_UTC"], row["END_DATETIME_UTC"],
                other["BEGIN_DATETIME_UTC"], other["END_DATETIME_UTC"],
                time_lag
            ):
                continue
            
            df.at[idx, "OVERLAPPING_EVENTS"].append(other["EVENT_ID"])
            
            if row["EVENT_TYPE"] != other["EVENT_TYPE"]:
                overlapping_event_pairs.append(((row["EVENT_ID"]), (other["EVENT_ID"])))
    
    df.loc[:, "OVERLAPPING_EVENTS"] = df["OVERLAPPING_EVENTS"].apply(
        lambda x: ",".join(map(str, x))
    )
    
    return df, overlapping_event_pairs


def create_multi_hazard_pairs(df, overlapping_event_pairs, pair_id_count):
    """Create multi-hazard event pairs with combined impacts."""
    overlapping_event_pairs_unique = unique_pairs(overlapping_event_pairs)
    all_pair_df = pd.DataFrame()
    
    if len(overlapping_event_pairs_unique) == 0:
        return all_pair_df, pair_id_count
    
    for i in range(len(overlapping_event_pairs_unique)):
        pair_df_1 = df[df["EVENT_ID"] == overlapping_event_pairs_unique[i][0]][0:1].copy()
        pair_df_2 = df[df["EVENT_ID"] == overlapping_event_pairs_unique[i][1]][0:1].copy()
        
        # Round coordinates
        for pair_df in [pair_df_1, pair_df_2]:
            pair_df["BEGIN_LAT"] = pair_df["BEGIN_LAT"].round(2)
            pair_df["BEGIN_LON"] = pair_df["BEGIN_LON"].round(2)
            pair_df["END_LAT"] = pair_df["END_LAT"].round(2)
            pair_df["END_LON"] = pair_df["END_LON"].round(2)
        
        pair_df_1["PAIR_ID"] = pair_id_count
        pair_df_2["PAIR_ID"] = pair_id_count
        pair_id_count += 1
        
        temp_df = pd.concat([pair_df_1, pair_df_2])
        temp_df = temp_df.sort_values(by="EVENT_TYPE")
        all_pair_df = pd.concat([all_pair_df, temp_df])
    
    all_pair_df = all_pair_df.drop_duplicates()
    
    return all_pair_df, pair_id_count


def add_combined_impact_metrics(all_pair_df):
    """Add combined impact metrics for multi-hazard pairs."""
    if len(all_pair_df) == 0:
        return all_pair_df
    
    impact_cols = {
        "MULTI_INJURIES_DIRECT": "INJURIES_DIRECT",
        "MULTI_INJURIES_INDIRECT": "INJURIES_INDIRECT",
        "MULTI_DEATHS_DIRECT": "DEATHS_DIRECT",
        "MULTI_DEATHS_INDIRECT": "DEATHS_INDIRECT",
        "MULTI_ADJ_DAMAGE_PROPERTY": "ADJ_DAMAGE_PROPERTY",
        "MULTI_ADJ_DAMAGE_CROPS": "ADJ_DAMAGE_CROPS",
    }
    
    for multi_col, single_col in impact_cols.items():
        all_pair_df[multi_col] = all_pair_df.groupby("PAIR_ID")[single_col].transform("sum")
    
    return all_pair_df


def reindex_pair_columns(df):
    """Reorder multi-hazard pair dataframe columns."""
    column_order = [
        "PAIR_ID", "OVERLAPPING_EVENTS", "EPISODE_ID", "EVENT_ID", "GEOID", "STATE",
        "STATE_FIPS", "EVENT_TYPE", "HAZARD", "CZ_TYPE", "CZ_FIPS", "CZ_NAME",
        "BEGIN_DATETIME_UTC", "END_DATETIME_UTC", "BEGIN_DATETIME", "END_DATETIME",
        "start_year", "end_year", "WFO", "CZ_TIMEZONE", "MULTI_INJURIES_DIRECT",
        "INJURIES_DIRECT", "MULTI_INJURIES_INDIRECT", "INJURIES_INDIRECT",
        "MULTI_DEATHS_DIRECT", "DEATHS_DIRECT", "MULTI_DEATHS_INDIRECT",
        "DEATHS_INDIRECT", "MULTI_ADJ_DAMAGE_PROPERTY", "ADJ_DAMAGE_PROPERTY",
        "MULTI_ADJ_DAMAGE_CROPS", "ADJ_DAMAGE_CROPS", "SOURCE", "MAGNITUDE",
        "MAGNITUDE_TYPE", "FLOOD_CAUSE", "CATEGORY", "TOR_F_SCALE", "TOR_LENGTH",
        "TOR_WIDTH", "TOR_OTHER_WFO", "TOR_OTHER_CZ_STATE", "TOR_OTHER_CZ_FIPS",
        "TOR_OTHER_CZ_NAME", "BEGIN_RANGE", "BEGIN_AZIMUTH", "BEGIN_LOCATION",
        "END_RANGE", "END_AZIMUTH", "END_LOCATION", "BEGIN_LAT", "BEGIN_LON",
        "END_LAT", "END_LON", "DATA_SOURCE", "EPISODE_NARRATIVE", "EVENT_NARRATIVE",
    ]
    return df.reindex(columns=column_order)


def identify_multi_hazard_pairs_by_state(ae_df, time_lag):
    """Iterate through states and counties to identify multi-hazard pairs."""
    mhp_df = pd.DataFrame()
    pair_id_count = 0
    
    state_fips_list = sorted(ae_df["STATE_FIPS"].unique().tolist())
    
    for state_fips in tqdm(state_fips_list, desc="Processing states"):
        state_df = ae_df[ae_df["STATE_FIPS"] == state_fips]
        county_fips_list = sorted(state_df["CZ_FIPS"].unique().tolist())
        
        for county_fips in county_fips_list:
            df = state_df[state_df["CZ_FIPS"] == county_fips].copy()
            
            df, overlapping_event_pairs = identify_overlapping_events(df, time_lag)
            all_pair_df, pair_id_count = create_multi_hazard_pairs(df, overlapping_event_pairs, pair_id_count)
            
            if len(all_pair_df) > 0:
                all_pair_df = add_combined_impact_metrics(all_pair_df)
                all_pair_df = reindex_pair_columns(all_pair_df)
                all_pair_df = all_pair_df.drop_duplicates()
                mhp_df = pd.concat([mhp_df, all_pair_df])
    
    return mhp_df.reset_index(drop=True)


def save_parquet(df, output_path):
    """Save dataframe to compressed parquet file."""
    df.to_parquet(output_path, compression="gzip")
    print(f"Saved: {output_path}")


def prepare_county_shapefiles(shapefile_path):
    """Load and prepare US county shapefile."""
    us_county_polygons = gpd.read_file(shapefile_path)
    us_county_polygons = us_county_polygons.dissolve(by='GEOID')
    us_county_polygons = us_county_polygons.reset_index(drop=False)
    us_county_polygons['GEOID'] = us_county_polygons['GEOID'].astype(str).str.zfill(5)
    print(f'US County Polygon CRS: {us_county_polygons.geometry.crs}')
    return us_county_polygons


def get_nested_dict_values(data):
    """Extract all values from nested dictionaries."""
    values = []
    for value in data.values():
        if isinstance(value, dict):
            values.extend(get_nested_dict_values(value))
        else:
            values.extend(value)
    return values


def build_county_event_dictionaries(ae_df, mhp_df, us_county_polygons, year_range):
    """Create nested dictionaries of county events by year/state/county."""
    sh_count_dict = {}
    sh_event_dict = {}
    mhp_count_dict = {}
    mhp_event_dict = {}
    no_hazard_boolean_dict = {}
    sh_boolean_dict = {}
    mhp_boolean_dict = {}
    no_hazard_or_sh_boolean_dict = {}
    sh_or_mhp_boolean_dict = {}
    
    state_list = sorted(us_county_polygons['STATEFP'].unique().tolist())
    
    for year in tqdm(year_range, desc="Processing years"):
        # Initialize year level
        sh_count_dict[year] = {}
        mhp_count_dict[year] = {}
        sh_event_dict[year] = {}
        mhp_event_dict[year] = {}
        no_hazard_boolean_dict[year] = {}
        sh_boolean_dict[year] = {}
        mhp_boolean_dict[year] = {}
        no_hazard_or_sh_boolean_dict[year] = {}
        sh_or_mhp_boolean_dict[year] = {}
        
        ae_sub = ae_df[(ae_df['start_year'] == year) | (ae_df['end_year'] == year)].reset_index(drop=True)
        mhp_sub = mhp_df[(mhp_df['start_year'] == year) | (mhp_df['end_year'] == year)].reset_index(drop=True)
        
        for state in state_list:
            # Initialize state level
            sh_count_dict[year][state] = {}
            mhp_count_dict[year][state] = {}
            sh_event_dict[year][state] = {}
            mhp_event_dict[year][state] = {}
            no_hazard_boolean_dict[year][state] = {}
            sh_boolean_dict[year][state] = {}
            mhp_boolean_dict[year][state] = {}
            no_hazard_or_sh_boolean_dict[year][state] = {}
            sh_or_mhp_boolean_dict[year][state] = {}
            
            county_state_list = us_county_polygons.loc[
                us_county_polygons['STATEFP'] == state, ['GEOID', 'COUNTYFP', 'geometry']
            ]
            
            for county in county_state_list['GEOID']:
                # Filter events by county
                ae_county = ae_sub[ae_sub['GEOID'] == county].reset_index(drop=True)
                mhp_county = mhp_sub[mhp_sub['GEOID'] == county].reset_index(drop=True)
                
                # Identify single-hazard only events (not part of multi-hazard pairs)
                multi_events = set(mhp_county['EVENT_ID'].unique())
                single_events = set(ae_county['EVENT_ID'].unique())
                single_only_events = single_events.symmetric_difference(multi_events)
                sh_only_county = ae_county[ae_county['EVENT_ID'].isin(single_only_events)]
                
                # Populate single-hazard dict
                if len(sh_only_county) > 0:
                    sh_boolean_dict[year][state][county] = True
                    sh_count_dict[year][state][county] = len(sh_only_county['EVENT_ID'].unique())
                    sh_event_dict[year][state][county] = sh_only_county['EVENT_ID'].unique().tolist()
                else:
                    sh_boolean_dict[year][state][county] = False
                    sh_count_dict[year][state][county] = 0
                    sh_event_dict[year][state][county] = []
                
                # Populate multi-hazard dict
                multi_duplicated_events = mhp_county["PAIR_ID"][mhp_county["PAIR_ID"].duplicated(keep=False)]
                multi_filtered_df = mhp_county[mhp_county["PAIR_ID"].isin(multi_duplicated_events)]
                
                if len(multi_filtered_df) > 0:
                    mhp_boolean_dict[year][state][county] = True
                    mhp_count_dict[year][state][county] = len(multi_filtered_df['PAIR_ID'].unique())
                    mhp_event_dict[year][state][county] = multi_filtered_df['PAIR_ID'].unique().tolist()
                else:
                    mhp_boolean_dict[year][state][county] = False
                    mhp_count_dict[year][state][county] = 0
                    mhp_event_dict[year][state][county] = []
                
                # Populate boolean dicts
                no_hazard_boolean_dict[year][state][county] = (len(sh_only_county) == 0) and (len(multi_filtered_df) == 0)
                no_hazard_or_sh_boolean_dict[year][state][county] = len(multi_filtered_df) == 0
                sh_or_mhp_boolean_dict[year][state][county] = (len(sh_only_county) > 0) or (len(multi_filtered_df) > 0)
    
    return {
        'sh_count': sh_count_dict,
        'sh_event': sh_event_dict,
        'mhp_count': mhp_count_dict,
        'mhp_event': mhp_event_dict,
        'no_hazard': no_hazard_boolean_dict,
        'sh_boolean': sh_boolean_dict,
        'mhp_boolean': mhp_boolean_dict,
        'no_hazard_or_sh': no_hazard_or_sh_boolean_dict,
        'sh_or_mhp': sh_or_mhp_boolean_dict,
    }


def save_dictionaries_as_pickle(dicts, output_path):
    """Save all dictionaries as pickle files."""
    dict_mapping = {
        'sh_count': 'NCEI_County_SH_count_dict.pkl',
        'sh_event': 'NCEI_County_SH_only_event_dict.pkl',
        'mhp_count': 'NCEI_County_MHP_count_dict.pkl',
        'mhp_event': 'NCEI_County_MHP_event_dict.pkl',
        'no_hazard': 'NCEI_County_NH_boolean_dict.pkl',
        'sh_boolean': 'NCEI_County_SH_boolean_dict.pkl',
        'mhp_boolean': 'NCEI_County_MHP_boolean_dict.pkl',
        'no_hazard_or_sh': 'NCEI_County_SH_NH_boolean_dict.pkl',
        'sh_or_mhp': 'NCEI_County_SH_MHP_boolean_dict.pkl',
    }
    
    for key, filename in dict_mapping.items():
        filepath = os.path.join(output_path, filename)
        with open(filepath, 'wb') as file:
            pickle.dump(dicts[key], file)
    
    print("All hazard dicts saved as pickle")


def save_summary_statistics(ae_df, sh_df, mhp_df, mh_df, output_path):
    """Save event set count statistics to text file."""
    event_set_count_outputs = [
        f"Total Number of Hazard Events: {len(ae_df)}",
        f"Number of Single-Hazard Only Events (SH): {len(sh_df)}",
        f"Number of Multi-Hazard Pairs (MHP): {int(len(mhp_df)/2)}",
        f"Number of Unique Multi-Hazard Events (MH): {len(mh_df)}",
    ]
    
    with open(output_path, "w", encoding="utf-8") as fp:
        for line in event_set_count_outputs:
            fp.write(line + "\n")
    
    print(f"Counts saved to: {output_path}")
    
    for line in event_set_count_outputs:
        print(line)


# ============================================================================
# MAIN FUNCTION
# ============================================================================

def generate_ncei_multihazard_eventset(
    cleaned_ncei_parquet_path,
    start_year,
    end_year,
    time_lag_days,
    hazard_event_inclusion_filter,
    inj=1,
    dth=1,
    c=50,
    p=50,
    us_county_shapefile_path=None,
    output_base_path=None,
):
    """
    Generate NCEI Storm Multihazard Event Sets from cleaned NCEI database.
    
    This function processes the cleaned NCEI Storm Database to create four output event sets:
    - AE: All Hazard Events (filtered by parameters)
    - SH: Single-Hazard Only Events (events not part of multi-hazard pairs)
    - MHP: Multi-Hazard Event Pairs (pairs of overlapping different hazards)
    - MH: Unique Multi-Hazard Events (deduplicated from MHP)
    
    Parameters
    ----------
    cleaned_ncei_parquet_path : str
        Path to cleaned NCEI Storm Database parquet file.
    start_year : int
        Start year for filtering events.
    end_year : int
        End year for filtering events.
    time_lag_days : int
        Temporal overlap window in days for identifying multi-hazard pairs.
    hazard_event_inclusion_filter : list
        List of hazard acronyms to include (e.g., ['tn', 'fl', 'hw']).
    inj : int, optional
        Minimum injuries threshold (default: 1).
    dth : int, optional
        Minimum deaths threshold (default: 1).
    c : int, optional
        Minimum crop damage threshold in thousands USD (default: 50).
    p : int, optional
        Minimum property damage threshold in thousands USD (default: 50).
    us_county_shapefile_path : str, optional
        Path to US county shapefile for county-level dictionary creation.
        If None, county dictionaries are not generated.
    output_base_path : str, optional
        Base output directory path. If None, uses current directory.
    
    Returns
    -------
    dict
        Dictionary containing:
        - 'ae': All Hazard Events dataframe
        - 'sh': Single-Hazard Only Events dataframe
        - 'mhp': Multi-Hazard Event Pairs dataframe
        - 'mh': Unique Multi-Hazard Events dataframe
        - 'ae_path': Path to AE parquet file
        - 'sh_path': Path to SH parquet file
        - 'mhp_path': Path to MHP parquet file
        - 'mh_path': Path to MH parquet file
    
    Example
    -------
    >>> result = generate_ncei_multihazard_eventset(
    ...     cleaned_ncei_parquet_path='path/to/cleaned_db.parquet',
    ...     start_year=1996,
    ...     end_year=2024,
    ...     time_lag_days=10,
    ...     hazard_event_inclusion_filter=['tn', 'fl', 'hw', 'tc'],
    ...     inj=1,
    ...     dth=1,
    ...     c=50,
    ...     p=50,
    ...     output_base_path='path/to/output'
    ... )
    """
    
    # Set up output directories
    if output_base_path is None:
        output_base_path = os.getcwd()
    
    create_folder_if_not_exists(output_base_path)
    
    output_dict_path = os.path.join(
        output_base_path,
        f'Eventset_Dicts_{inj}inj_{dth}dth_{c}c_{p}p_lag{time_lag_days}_{start_year}-{end_year}'
    )
    create_folder_if_not_exists(output_dict_path)
    
    time_lag = pd.Timedelta(days=time_lag_days)
    year_range = range(start_year, end_year + 1)
    
    # Load and prepare data
    print("\n--- Loading and Preparing Data ---")
    ae_df = load_and_prepare_data(cleaned_ncei_parquet_path)
    ae_df = apply_qa_qc_checks(ae_df)
    ae_df = clean_impact_columns(ae_df)
    ae_df = reindex_columns(ae_df)
    ae_df = filter_by_temporal_range(ae_df, start_year, end_year)
    ae_df = filter_by_state_and_zone(ae_df)
    ae_df = filter_by_hazard_type(ae_df, hazard_event_inclusion_filter)
    ae_df = filter_by_impact(ae_df, inj, dth, c, p)
    ae_df = ae_df.drop_duplicates()
    
    print(f"Total events after filtering: {len(ae_df)}")
    
    # Save All Events
    print("\n--- Saving All Events (AE) ---")
    ae_path = os.path.join(
        output_base_path,
        f'AE_{inj}inj_{dth}dth_{c}c_{p}p_lag{time_lag_days}_{start_year}-{end_year}.parquet.gz'
    )
    save_parquet(ae_df, ae_path)
    
    # Disable pandas chained assignment warning
    pd.options.mode.chained_assignment = None
    
    # Identify multi-hazard pairs
    print("\n--- Identifying Multi-Hazard Pairs ---")
    mhp_df = identify_multi_hazard_pairs_by_state(ae_df, time_lag)
    
    print(f"Total multi-hazard pairs: {int(len(mhp_df)/2)}")
    
    # Save Multi-Hazard Pairs
    print("\n--- Saving Multi-Hazard Pairs (MHP) ---")
    mhp_path = os.path.join(
        output_base_path,
        f'MHP_{inj}inj_{dth}dth_{c}c_{p}p_lag{time_lag_days}_{start_year}-{end_year}.parquet.gz'
    )
    save_parquet(mhp_df, mhp_path)
    
    # Create single-hazard only events
    print("\n--- Creating Single-Hazard Only Events (SH) ---")
    if us_county_shapefile_path is not None:
        us_county_polygons = prepare_county_shapefiles(us_county_shapefile_path)
        county_dicts = build_county_event_dictionaries(ae_df, mhp_df, us_county_polygons, year_range)
        sh_event_ids = get_nested_dict_values(county_dicts['sh_event'])
        sh_df = ae_df[ae_df['EVENT_ID'].isin(sh_event_ids)].reset_index(drop=True)
        
        # Save county dictionaries
        print("\n--- Saving County Event Dictionaries ---")
        save_dictionaries_as_pickle(county_dicts, output_dict_path)
    else:
        # If no shapefile provided, derive SH from events not in MHP
        mhp_event_ids = set(mhp_df['EVENT_ID'].unique())
        all_event_ids = set(ae_df['EVENT_ID'].unique())
        sh_event_ids = all_event_ids - mhp_event_ids
        sh_df = ae_df[ae_df['EVENT_ID'].isin(sh_event_ids)].reset_index(drop=True)
    
    print(f"Single-hazard only events: {len(sh_df)}")
    
    # Save Single-Hazard Only Events
    sh_path = os.path.join(
        output_base_path,
        f'SH_{inj}inj_{dth}dth_{c}c_{p}p_lag{time_lag_days}_{start_year}-{end_year}.parquet.gz'
    )
    save_parquet(sh_df, sh_path)
    
    # Create unique multi-hazard events
    print("\n--- Creating Unique Multi-Hazard Events (MH) ---")
    mh_df = mhp_df.drop_duplicates(subset='EVENT_ID', keep='first', ignore_index=True).reset_index(drop=True)
    
    print(f"Unique multi-hazard events: {len(mh_df)}")
    
    # Save Unique Multi-Hazard Events
    mh_path = os.path.join(
        output_base_path,
        f'MH_{inj}inj_{dth}dth_{c}c_{p}p_lag{time_lag_days}_{start_year}-{end_year}.parquet.gz'
    )
    save_parquet(mh_df, mh_path)
    
    # Save summary statistics
    print("\n--- Summary Statistics ---")
    summary_path = os.path.join(
        output_base_path,
        f'eventset_counts_{inj}inj_{dth}dth_{c}c_{p}p_lag{time_lag_days}_{start_year}-{end_year}.txt'
    )
    save_summary_statistics(ae_df, sh_df, mhp_df, mh_df, summary_path)
    
    print("\n--- Processing Complete ---\n")
    
    return {
        'ae': ae_df,
        'sh': sh_df,
        'mhp': mhp_df,
        'mh': mh_df,
        'ae_path': ae_path,
        'sh_path': sh_path,
        'mhp_path': mhp_path,
        'mh_path': mh_path,
    }


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    # User-defined parameters
    CLEANED_NCEI_PARQUET_PATH = r"C:\path\to\cleaned\NCEI_Storm_Database_Cleaned_Details_1996-2024_v2.parquet"
    US_COUNTY_SHAPEFILE_PATH = r'C:\path\to\CONUS_cb_2018_us_county_500k_WGS84.shp'
    OUTPUT_BASE_PATH = r'C:\output_path'
    
    START_YEAR = 1996
    END_YEAR = 2024
    TIME_LAG_DAYS = 10
    
    HAZARD_INCLUSION_FILTER = [
        "av", "bz", "cfl", "cw", "df", "dr", "ds", "ew", "fc", "ff", "fg", "fl",
        "hl", "hs", "hw", "is", "lt", "ltn", "rn", "pfl", "rc", "se", "sl", "sm",
        "sn", "sst", "swv", "tc", "tn", "ts", "tw", "vo", "wf", "wp", "ws", "ww"
    ]
    
    IMPACT_THRESHOLDS = {
        'inj': 1,
        'dth': 1,
        'c': 50,
        'p': 50,
    }
    
    # Run the main function
    result = generate_ncei_multihazard_eventset(
        cleaned_ncei_parquet_path=CLEANED_NCEI_PARQUET_PATH,
        start_year=START_YEAR,
        end_year=END_YEAR,
        time_lag_days=TIME_LAG_DAYS,
        hazard_event_inclusion_filter=HAZARD_INCLUSION_FILTER,
        inj=IMPACT_THRESHOLDS['inj'],
        dth=IMPACT_THRESHOLDS['dth'],
        c=IMPACT_THRESHOLDS['c'],
        p=IMPACT_THRESHOLDS['p'],
        us_county_shapefile_path=US_COUNTY_SHAPEFILE_PATH,
        output_base_path=OUTPUT_BASE_PATH,
    )
