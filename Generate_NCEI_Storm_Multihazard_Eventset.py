#######################
"""
Created 2025
@author: Joshua Green - University of Southampton

Please cite this script/dataset if used in any publications:

Green, J. (2025) NCEI Storm Multihazard Eventset.

This script uses the cleaned NCEI Storm Database to create four output events sets
- (AE) All Hazard Events
- (SH) Single-Hazard Only Events
- (MHP) Multi-Hazard Event Pairs
- (MH) Unique Multi-Hazard Events

User defined parameters allow for customization of event sets 
- eventset duration (start/end year),
- hazard type (list of hazard acronyms)
- multi-hazard time lag (temporal overlap period in days)
- minimum hazard impact filter thresholds (injury, death, building damage, and crop damage)

"""
#######################

import os, datetime
from datetime import datetime
import pandas as pd
import geopandas as gpd
import numpy as np
from tqdm import tqdm
import pickle

#pd.set_option('display.max_colwidth', None)
#pd.set_option('display.max_columns', None)

######################################################################################################
#                        USER DEFINED PARAMETERS
######################################################################################################
Cleaned_NCEI_Storm_Database_Parquet_Path = r"C:\Users\jg2n22\OneDrive - University of Southampton\Data\Hazards_Disasters\NCEI_Storm_Database_Cleaned\NCEI_Storm_Database_Cleaned_Details_1996-2024_v2.parquet"
#Hazard_Eventset_Output_Path = r"C:\Users\jg2n22\OneDrive - University of Southampton\Data\Hazards_Disasters\Updated_NCEI_Eventsets_v3"
Hazard_Eventset_Output_Path = r'C:\test_mh_output'
US_County_Shapefile_Path = r'C:\Users\jg2n22\OneDrive - University of Southampton\Data\Admin_Bounds\CONUS_cb_2018_us_county_500k\CONUS_cb_2018_us_county_500k_WGS84.shp'

# Define temporal year range
# CHANGE THESE VALUES AS DESIRED FOR TEMPORAL COVERAGE
start_year = 1996
end_year = 2024
year_range = range(start_year, end_year+2, 1)


# Define time lag in days
# CHANGE THESE VALUES AS DESIRED FOR APPROPRIATE TEMPORAL OVERLAP
#time_lag_days = 90
#time_lag_days = 30
time_lag_days = 10
time_lag = pd.Timedelta(days=time_lag_days)
time_lag_int = time_lag.days

# Define which hazard event types to include in the database, see lookup table in comments below
# CHANGE THESE VALUES AS DESIRED FOR HAZARD/PERIL TYPE    
hazard_event_inclusion_filter = ["av","bz","cfl","cw","df","dr","ds","ew","fc","ff","fg","fl","hl","hs","hw","is","lt","ltn","rn","pfl","rc","se","sl","sm","sn","sst","swv","tc","tn","ts","tw","vo","wf","wp","ws","ww"]
hazard_event_exclusion_filter = ["nl","mew","mtw","mhl","mfg","mtc","mltn"]

# Impact filter thresholds, minimum values for including in final event set
# CHANGE THESE VALUES AS DESIRED FOR APPROPRIATE IMPACT FILTERING
inj = 1 # injuries
dth = 1 # deaths
c = 50  # crop damage in thousands
p = 50  # property damage in thousands
#c = 10  # crop damage in thousands
#p = 10  # property damage in thousands

######################################################################################################

Hazard_Dict_Output_Path = rf'{Hazard_Eventset_Output_Path}\Eventset_Dicts_{inj}inj_{dth}dth_{c}c_{p}p_lag{time_lag_days}_{start_year}-{end_year}'


def create_folder_if_not_exists(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"Folder created at: {folder_path}")
    else:
        print(f"Folder already exists at: {folder_path}")
create_folder_if_not_exists(Hazard_Eventset_Output_Path)
create_folder_if_not_exists(Hazard_Dict_Output_Path)


# Load the cleaned NCEI storm database
raw_df = pd.read_parquet(Cleaned_NCEI_Storm_Database_Parquet_Path)
AE = raw_df

AE["CZ_FIPS"] = AE["CZ_FIPS"].astype(str).str.zfill(3)
AE["STATE_FIPS"] = AE["STATE_FIPS"].astype(str).str.zfill(2)
AE["GEOID"] = AE["STATE_FIPS"].astype(str).str.zfill(2) + AE[
    "CZ_FIPS"
].astype(str).str.zfill(3)


# General qa/qc. Problems should have been removed during database cleaning, however complete additional final check.
#AE = AE[~AE["EPISODE_ID"].isnull()]
AE = AE[~AE["EVENT_ID"].isnull()]
AE = AE[~AE["STATE"].isnull()]
AE = AE[~AE["STATE_FIPS"].isnull()]
AE = AE[~AE["EVENT_TYPE"].isnull()]
AE = AE[~AE["CZ_FIPS"].isnull()]
AE = AE[~AE["BEGIN_DATETIME_UTC"].isnull()]
AE = AE[~AE["END_DATETIME_UTC"].isnull()]
AE = AE[~AE["BEGIN_DATETIME"].isnull()]
AE = AE[~AE["END_DATETIME"].isnull()]

# Uncomment to remove these descriptive columns if desired
# AE = AE.drop(columns=['EPISODE_NARRATIVE', 'EVENT_NARRATIVE'])

# Define which hazard event types to include in the single/multi-hazard event sets
# "av":"Avalanche"
# "bz":"Blizzard"
# "cfl":"Coastal Flood"
# "cw":"Cold/Wind Chill"
# "df":"Debris Flow"
# "dr":"Drought"
# "ds":"Dust Storm"
# "ew":"Extreme Wind"
# "fc":"Funnel Cloud"
# "ff":"Frost/Freeze"
# "fg":"Fog"
# "fl":"Flood"
# "hl":"Hail"
# "hs":"High Surf"
# "hw":"Heat"
# "is":"Ice Storm"
# "lt":"Astronomical Low Tide"
# "ltn":"Lightning"
# "mew":"Marine Extreme Wind"
# "mfg":"Marine Fog"
# "mhl":"Marine Hail"
# "mht":"Marine Tropical Cyclone"
# "mltn":"Marine Lightning"
# "mtw":"Marine Thunderstorm Wind"
# "nl":"Northern Lights"
# "rn":"Rain"
# "pfl":"Flash Flood"
# "rc":"Rip Current"
# "se":"Seiche"
# "sl":"Sleet"
# "sm":"Smoke"
# "sn":"Snow"
# "sst":"Storm Surge/Tide"
# "swv":"Sneakerwave"
# "tc":"Tropical Cyclone"
# "tn":"Tornado"
# "ts":"Tsunami"
# "tw":"Thunderstorm Wind"
# "vo":"Volcanic Ash"
# "wf":"Wildfire"
# "wp":"Waterspout"
# "ws":"Winter Storm"
# "ww":"Winter Weather"

# Subset database to only desired event types
AE = AE[AE['HAZARD'].isin(hazard_event_inclusion_filter)]


# General qa/qc and preprocessing
AE['DEATHS_DIRECT'] = AE['DEATHS_DIRECT'].fillna(0).astype(int)
AE['DEATHS_INDIRECT'] = AE['DEATHS_INDIRECT'].fillna(0).astype(int)
AE['INJURIES_DIRECT'] = AE['INJURIES_DIRECT'].fillna(0).astype(int)
AE['INJURIES_DIRECT'] = AE['INJURIES_DIRECT'].fillna(0).astype(int)
AE["DAMAGE_CROPS"] = AE["DAMAGE_CROPS"].fillna(0).astype(int)
AE["DAMAGE_PROPERTY"] = AE["DAMAGE_PROPERTY"].fillna(0).astype(int)
AE['ADJ_DAMAGE_CROPS'] = AE['ADJ_DAMAGE_CROPS'].fillna(0).astype(int)
AE['ADJ_DAMAGE_PROPERTY'] = AE['ADJ_DAMAGE_PROPERTY'].fillna(0).astype(int)
AE['TOTAL_INJURIES'] = AE['TOTAL_INJURIES'].fillna(0)
AE['TOTAL_DEATHS'] = AE['TOTAL_DEATHS'].fillna(0)
AE['TOTAL_ADJ_DAMAGE'] = AE['TOTAL_ADJ_DAMAGE'].fillna(0)

AE.loc[AE['DEATHS_DIRECT']<0, 'DEATHS_DIRECT'] = 0
AE.loc[AE['DEATHS_INDIRECT']<0, 'DEATHS_INDIRECT'] = 0
AE.loc[AE['INJURIES_DIRECT']<0, 'INJURIES_DIRECT'] = 0
AE.loc[AE['INJURIES_DIRECT']<0, 'INJURIES_DIRECT'] = 0
AE.loc[AE['DAMAGE_CROPS']<0, 'DAMAGE_CROPS'] = 0
AE.loc[AE['DAMAGE_PROPERTY']<0, 'DAMAGE_PROPERTY'] = 0
AE.loc[AE['ADJ_DAMAGE_CROPS']<0, 'ADJ_DAMAGE_CROPS'] = 0
AE.loc[AE['ADJ_DAMAGE_PROPERTY']<0, 'ADJ_DAMAGE_PROPERTY'] = 0
AE.loc[AE['TOTAL_INJURIES']<0, 'TOTAL_INJURIES'] = 0
AE.loc[AE['TOTAL_DEATHS']<0, 'TOTAL_DEATHS'] = 0
AE.loc[AE['TOTAL_ADJ_DAMAGE']<0, 'TOTAL_ADJ_DAMAGE'] = 0

AE = AE.drop_duplicates()


AE = AE.reindex(
    columns=[
        "EPISODE_ID",
        "EVENT_ID",
        "GEOID",
        "STATE",
        "STATE_FIPS",
        "EVENT_TYPE",
        "HAZARD",
        "CZ_TYPE",
        "CZ_FIPS",
        "CZ_NAME",
        "BEGIN_DATETIME_UTC",
        "END_DATETIME_UTC",
        "BEGIN_DATETIME",
        "END_DATETIME",
        "start_year",
        "end_year",
        "WFO",
        "CZ_TIMEZONE",
        "INJURIES_DIRECT",
        "INJURIES_INDIRECT",
        "DEATHS_DIRECT",
        "DEATHS_INDIRECT",
        "DAMAGE_PROPERTY",
        "DAMAGE_CROPS",
        "ADJ_DAMAGE_PROPERTY",
        "ADJ_DAMAGE_CROPS",
        'TOTAL_INJURIES',
        'TOTAL_DEATHS',
        'TOTAL_ADJ_DAMAGE',
        "SOURCE",
        "MAGNITUDE",
        "MAGNITUDE_TYPE",
        "FLOOD_CAUSE",
        "CATEGORY",
        "TOR_F_SCALE",
        "TOR_LENGTH",
        "TOR_WIDTH",
        "TOR_OTHER_WFO",
        "TOR_OTHER_CZ_STATE",
        "TOR_OTHER_CZ_FIPS",
        "TOR_OTHER_CZ_NAME",
        "BEGIN_RANGE",
        "BEGIN_AZIMUTH",
        "BEGIN_LOCATION",
        "END_RANGE",
        "END_AZIMUTH",
        "END_LOCATION",
        "BEGIN_LAT",
        "BEGIN_LON",
        "END_LAT",
        "END_LON",
        "DATA_SOURCE",
        "EPISODE_NARRATIVE",
        "EVENT_NARRATIVE",
    ]
)

# temporal filter
# Make sure using desired datetime, use UTC for standardized comparison, otherwise might be comparing two datetimes with different timezones
AE = AE[
    (AE["BEGIN_DATETIME_UTC"].dt.year >= start_year)
    & (AE["END_DATETIME_UTC"].dt.year <= end_year)
    # (AE["BEGIN_DATETIME"].dt.year >= start_year)
    # & (AE["END_DATETIME"].dt.year <= end_year)
    
]



# Remove unwanted state classes
# CHANGE THE EXCLUSED STATES AS DESIRED
# NOTE THAT THIS CLASSIFICATION INCLUDES US TERRITORIES AND WATER BODIES
Exclusion_State_List = [
    "ALASKA",
    "AMERICAN SAMOA",
    "ATLANTIC NORTH",
    "ATLANTIC SOUTH",
    "E PACIFIC",
    "GUAM WATERS",
    "GUAM",    
    "GULF OF ALASKA",
    "GULF OF MEXICO",
    "HAWAII WATERS",
    "HAWAII",
    "LAKE ERIE",
    "LAKE HURON",
    "LAKE MICHIGAN",
    "LAKE ONTARIO",
    "LAKE ST CLAIR",
    "LAKE SUPERIOR",
    "PUERTO RICO",
    "ST LAWRENCE R",
    "VIRGIN ISLANDS",
]
AE = AE[~AE["STATE"].isin(Exclusion_State_List)]


#remove marine zone only events, this should have been done as a byproduct of the above step, however this check is implemented as a backup
AE = AE[(AE['CZ_TYPE']!='M')]


# Filter by event impact
# Modify below to filter by 'ALL_INJURIES','ALL_DEATHS','TOTAL_ADJ_DAMAGE' if desired
AE = AE[
    (
        (AE["INJURIES_DIRECT"] >= inj)
        | (AE["INJURIES_INDIRECT"] >= inj)
        | (AE["DEATHS_DIRECT"] >= dth)
        | (AE["DEATHS_INDIRECT"] >= dth)
        | (AE["ADJ_DAMAGE_CROPS"] >= c * 1000)
        | (AE["ADJ_DAMAGE_PROPERTY"] >= p * 1000)
    )
]

# save prepared singledf, as a csv and/or parquet
# AE.to_csv(
#     rf"{Hazard_Eventset_Output_Path}\AE_{inj}inj_{dth}dth_{c}c_{p}p_lag{time_lag_int}_{start_year}-{end_year}.csv.gz",
#     compression="gzip",
#     encoding="utf-8",
#     index=False,
# )
AE.to_parquet(
    rf"{Hazard_Eventset_Output_Path}\AE_{inj}inj_{dth}dth_{c}c_{p}p_lag{time_lag_int}_{start_year}-{end_year}.parquet.gz",
    compression="gzip",
)

##CHECK WARNING####
pd.options.mode.chained_assignment = None  # default='warn'

# Define empty dataframes that will store the single hazard and multi-hazard eventsets
MHP = pd.DataFrame()
SH = pd.DataFrame()

# Define a list of the state fips codes
state_fips_list = AE["STATE_FIPS"].unique().tolist()
state_fips_list.sort()

# Record any counties that don't have any overlapping hazard events for the defined time lag
#No_MHP_County_df = pd.DataFrame()


# Check if datetime ranges overlap with a time lag
def datetime_ranges_overlap_with_lag(start1, end1, start2, end2, lag):
    return max(start1 - lag, start2 - lag) <= min(end1 + lag, end2 + lag)


# Check to make sure that marine events can only be paired with other marine events
def cz_types_compatible(ct1, ct2):
    # both Z/C or both M
    return ((ct1 in ("Z", "C") and ct2 in ("Z", "C"))
            or (ct1 == "M" and ct2 == "M"))
    
def unique_pairs(pairs):
    unique_set = set()
    unique_list = []

    for pair in pairs:
        sorted_pair = tuple(sorted(pair))
        if sorted_pair not in unique_set:
            unique_set.add(sorted_pair)
            unique_list.append(pair)

    return unique_list


# Define a function to combine values in a column
def combine_values_comma(values):
    return ",".join(map(str, values))


def combine_values_slash(values):
    return "/".join(map(str, values))


def check_combine_values_comma(x):
    # Check if all values in the group are the same
    if (x == x.iloc[0]).all():
        return x.iloc[0]
    else:
        return ",".join(
            map(str, x)
        )  # If different, combine as a string with a delimiter


def check_combine_values_slash(x):
    # Check if all values in the group are the same
    if (x == x.iloc[0]).all():
        return x.iloc[0]
    else:
        return "/".join(
            map(str, x)
        )  # If different, combine as a string with a delimiter


def combine_multi(x):
    v1, v2 = x.iloc[0], x.iloc[1]
    v1_is_number = pd.to_numeric(v1, errors="coerce")
    v2_is_number = pd.to_numeric(v2, errors="coerce")

    if pd.notnull(v1_is_number) and pd.notnull(v2_is_number):
        return v1_is_number + v2_is_number
    elif pd.notnull(v1_is_number):
        return v1_is_number
    elif pd.notnull(v2_is_number):
        return v2_is_number
    else:
        return 0

# Set counter used to assign multihazard pair ids
pair_id_count = 0


for state_fips in tqdm(state_fips_list):
    
    all_combined_pair_df = pd.DataFrame()
    
    #print(f"state_fips:{state_fips}")
    state_df = AE[AE["STATE_FIPS"] == state_fips]
    county_fips_list = sorted(state_df["CZ_FIPS"].unique().tolist())

    #for county_fips in tqdm(county_fips_list):
    for county_fips in county_fips_list:
        #print(f"state_fips:{state_fips}, county_fips:{county_fips}")

        df = state_df[state_df["CZ_FIPS"] == county_fips]

        # # Create DataFrame
        # df = pd.DataFrame(data)

        # Initialize the 'overlapping_events' column
        #df.loc[:, "OVERLAPPING_EVENTS"] = [[] for _ in range(len(df))]
        df["OVERLAPPING_EVENTS"] = [[] for _ in range(len(df))]

        # List to store pairs of events that overlap with different event types
        overlapping_event_pairs = []

        # # Iterate over each row in the DataFrame
        # for idx, row in df.iterrows():
        #     # Subset of rows with the same county location name and id
        #     subset = df[
        #         (df["CZ_FIPS"] == row["CZ_FIPS"])
        #         & (df["CZ_NAME"] == row["CZ_NAME"])
        #         & (df["STATE_FIPS"] == row["STATE_FIPS"])
        #     ]

        #     # Identify overlapping events with time lag
        #     for _, other_row in subset.iterrows():
        #         # Check to make sure that the event is not paired with itself, i.e. the event ids are different
        #         # Check that events overlap in time
        #         # Check
        #         if (row["EVENT_ID"] != other_row["EVENT_ID"] 
        #         and datetime_ranges_overlap_with_lag( 
        #             row["BEGIN_DATETIME"],
        #             row["END_DATETIME"],
        #             other_row["BEGIN_DATETIME"],
        #             other_row["END_DATETIME"],
        #             time_lag)
        #         and (cz_types_compatible(
        #             row["CZ_TYPE"],
        #             other["CZ_TYPE"])
        #             ):
                    
        #             df.at[idx, "OVERLAPPING_EVENTS"].append((other_row["EVENT_ID"])) 
        #             # Check to see if events are not the same type, excluding the pairing of the like-type events, 
        #             # Some events are recorded in parts despite being the same event, this avoids paring an event with itself
        #             if row["EVENT_TYPE"] != other_row["EVENT_TYPE"]:
        #                 overlapping_event_pairs.append(
        #                     ((row["EVENT_ID"]), (other_row["EVENT_ID"]))
        #                 )
        
        # Iterate over each row in the DataFrame
        for idx, row in df.iterrows():
            # Subset of rows with the same county location name and id
            subset = df[
                (df["CZ_FIPS"] == row["CZ_FIPS"]) &
                (df["CZ_NAME"] == row["CZ_NAME"]) &
                (df["STATE_FIPS"] == row["STATE_FIPS"])
            ]
            for _, other in subset.iterrows():
                #Check to make sure that the event is not paired with itself, if true skip iteration
                # Note that events with the same EPISODE_ID (i.e. storm episode) are allowed, just not the same individual storm event
                if row["EVENT_ID"] == other["EVENT_ID"]:
                    continue

                # Check if there are temporally overlapping events, if false skip iteration
                if not datetime_ranges_overlap_with_lag(
                        row["BEGIN_DATETIME_UTC"], row["END_DATETIME_UTC"],
                        other["BEGIN_DATETIME_UTC"], other["END_DATETIME_UTC"],
                        time_lag):
                    continue

                # UNCOMMENT IF DESIRED
                # Check if the overlapping events satisfy the CZ_TYPE pair rules, 'M' can only be paired with 'M', if false skip iteration
                #if not cz_types_compatible(row["CZ_TYPE"], other["CZ_TYPE"]):
                #    continue

                df.at[idx, "OVERLAPPING_EVENTS"].append(other["EVENT_ID"])

                # Check if the hazard event types are different (to avoid self-duplication), if true add event pairs to pair dataframe
                if row["EVENT_TYPE"] != other["EVENT_TYPE"]:
                    
                    overlapping_event_pairs.append(((row["EVENT_ID"]), (other["EVENT_ID"])))

        # Convert list to string for easier reading
        # df['OVERLAPPING_EVENTS'] = df['OVERLAPPING_EVENTS'].apply(lambda x: ', '.join(map(str, x)))
        df.loc[:, "OVERLAPPING_EVENTS"] = df["OVERLAPPING_EVENTS"].apply(
            lambda x: ",".join(map(str, x))
        )

        df = df[
            ["OVERLAPPING_EVENTS"]
            + [col for col in df.columns if col != "OVERLAPPING_EVENTS"]
        ]

        # Print the DataFrame to verify the result

        # print(df)
        # print("Pairs of overlapping events with different event types:")
        # print(overlapping_event_pairs)

        overlapping_event_pairs_unique = unique_pairs(overlapping_event_pairs)
        # print(len(overlapping_event_pairs_unique))

        all_pair_df = pd.DataFrame()

        # Check to make sure there are some overlapping events, if not then skip to the next county iteration
        if len(overlapping_event_pairs_unique) > 0:

            for i in range(0, len(overlapping_event_pairs_unique)):
                pair_df_1 = df[df["EVENT_ID"] == overlapping_event_pairs_unique[i][0]][
                    0:1
                ]
                pair_df_1["BEGIN_LAT"] = pair_df_1["BEGIN_LAT"].round(2)
                pair_df_1["BEGIN_LON"] = pair_df_1["BEGIN_LON"].round(2)
                pair_df_1["END_LAT"] = pair_df_1["END_LAT"].round(2)
                pair_df_1["END_LON"] = pair_df_1["END_LON"].round(2)
                # pair_df_1 = pair_df_1.drop_duplicates() #remove any duplicates, there should only be 1 unique df row here
                # print(len(pair_df_1))

                pair_df_2 = df[df["EVENT_ID"] == overlapping_event_pairs_unique[i][1]][0:1]
                pair_df_2["BEGIN_LAT"] = pair_df_2["BEGIN_LAT"].round(2)
                pair_df_2["BEGIN_LON"] = pair_df_2["BEGIN_LON"].round(2)
                pair_df_2["END_LAT"] = pair_df_2["END_LAT"].round(2)
                pair_df_2["END_LON"] = pair_df_2["END_LON"].round(2)
                # pair_df_2 = pair_df_2.drop_duplicates() #remove any duplicates, there should only be 1 unique df row here
                # print(len(pair_df_1))

                pair_df_1["PAIR_ID"] = pair_id_count
                pair_df_2["PAIR_ID"] = pair_id_count

                # pair_df_1['PAIR_ID'] = i
                # pair_df_2['PAIR_ID'] = i

                pair_id_count = pair_id_count + 1

                temp_df = pd.concat([pair_df_1, pair_df_2])
                temp_df = temp_df.sort_values(
                    by="EVENT_TYPE"
                )  # Reorder the two event pairs such that the event_type pairs are later formatted the same, when combined into a single string
                all_pair_df = pd.concat([all_pair_df, temp_df])
                all_pair_df = all_pair_df.drop_duplicates()

            all_pair_df = all_pair_df[
                ["PAIR_ID"] + [col for col in all_pair_df.columns if col != "PAIR_ID"]
            ]
            all_pair_df = all_pair_df.drop_duplicates()

            MULTI_INJURIES_DIRECT = all_pair_df.groupby("PAIR_ID")[
                "INJURIES_DIRECT"
            ].transform("sum")
            MULTI_INJURIES_INDIRECT = all_pair_df.groupby("PAIR_ID")[
                "INJURIES_INDIRECT"
            ].transform("sum")
            MULTI_DEATHS_DIRECT = all_pair_df.groupby("PAIR_ID")[
                "DEATHS_DIRECT"
            ].transform("sum")
            MULTI_DEATHS_INDIRECT = all_pair_df.groupby("PAIR_ID")[
                "DEATHS_INDIRECT"
            ].transform("sum")
            MULTI_DAMAGE_PROPERTY = all_pair_df.groupby("PAIR_ID")[
                "ADJ_DAMAGE_PROPERTY"
            ].transform("sum")
            MULTI_DAMAGE_CROPS = all_pair_df.groupby("PAIR_ID")[
                "ADJ_DAMAGE_CROPS"
            ].transform("sum")

            all_pair_df["MULTI_INJURIES_DIRECT"] = MULTI_INJURIES_DIRECT
            all_pair_df["MULTI_INJURIES_INDIRECT"] = MULTI_INJURIES_INDIRECT
            all_pair_df["MULTI_DEATHS_DIRECT"] = MULTI_DEATHS_DIRECT
            all_pair_df["MULTI_DEATHS_INDIRECT"] = MULTI_DEATHS_INDIRECT
            all_pair_df["MULTI_ADJ_DAMAGE_PROPERTY"] = MULTI_DAMAGE_PROPERTY
            all_pair_df["MULTI_ADJ_DAMAGE_CROPS"] = MULTI_DAMAGE_CROPS

            all_pair_df = all_pair_df.reindex(
                columns=[
                    "PAIR_ID",
                    "OVERLAPPING_EVENTS",
                    "EPISODE_ID",
                    "EVENT_ID",
                    "GEOID",
                    "STATE",
                    "STATE_FIPS",
                    "EVENT_TYPE",
                    "HAZARD",
                    "CZ_TYPE",
                    "CZ_FIPS",
                    "CZ_NAME",
                    "BEGIN_DATETIME_UTC",
                    "END_DATETIME_UTC",
                    "BEGIN_DATETIME",
                    "END_DATETIME",
                    "start_year",
                    "end_year",
                    "WFO",
                    "CZ_TIMEZONE",
                    "MULTI_INJURIES_DIRECT",
                    "INJURIES_DIRECT",
                    "MULTI_INJURIES_INDIRECT",
                    "INJURIES_INDIRECT",
                    "MULTI_DEATHS_DIRECT",
                    "DEATHS_DIRECT",
                    "MULTI_DEATHS_INDIRECT",
                    "DEATHS_INDIRECT",
                    "MULTI_ADJ_DAMAGE_PROPERTY",
                    "ADJ_DAMAGE_PROPERTY",
                    "MULTI_ADJ_DAMAGE_CROPS",
                    "ADJ_DAMAGE_CROPS",
                    "SOURCE",
                    "MAGNITUDE",
                    "MAGNITUDE_TYPE",
                    "FLOOD_CAUSE",
                    "CATEGORY",
                    "TOR_F_SCALE",
                    "TOR_LENGTH",
                    "TOR_WIDTH",
                    "TOR_OTHER_WFO",
                    "TOR_OTHER_CZ_STATE",
                    "TOR_OTHER_CZ_FIPS",
                    "TOR_OTHER_CZ_NAME",
                    "BEGIN_RANGE",
                    "BEGIN_AZIMUTH",
                    "BEGIN_LOCATION",
                    "END_RANGE",
                    "END_AZIMUTH",
                    "END_LOCATION",
                    "BEGIN_LAT",
                    "BEGIN_LON",
                    "END_LAT",
                    "END_LON",
                    "DATA_SOURCE",
                    "EPISODE_NARRATIVE",
                    "EVENT_NARRATIVE",
                ]
            )

            all_combined_pair_df = pd.concat([all_combined_pair_df,all_pair_df])
            MHP = pd.concat([MHP, all_pair_df])



# Load us county shapefile, used to complete spatial filtering, can implement via shapely.STRtree() 
us_county_polygons = gpd.read_file(US_County_Shapefile_Path)
us_county_polygons = us_county_polygons.dissolve(by='GEOID')# some of the counties have multiple small polygons, dissolve into one single polygon
us_county_polygons = us_county_polygons.reset_index(drop=False)
#us_county_polygons['GEOID'] = us_county_polygons['GEOID'].astype(int).astype(str) #remove leading zeros
us_county_polygons['GEOID'] = us_county_polygons['GEOID'].astype(str).str.zfill(5) #add leading zeros

print(f'US County Polygon CRS: {us_county_polygons.geometry.crs}')

# Define dictionaries that will store county event info, in a 3x nested structure of year->state->county
SH_count_dict = {}
SH_event_dict = {}

MHP_count_dict = {}
MHP_event_dict = {}

no_hazard_boolean_dict = {}
SH_boolean_dict = {}
MHP_boolean_dict = {}
no_hazard_or_SH_boolean_dict = {}
SH_or_MHP_boolean_dict = {}

# Define list of states to iterate through
state_list = us_county_polygons['STATEFP'].unique().tolist() 


# Iterate through the previous defined start/end years
for year in tqdm(year_range):
    #print(f'Year: {year}')
    # Define nested structure of dictionaries
    SH_count_dict[year] = {}
    MHP_count_dict[year] = {}
    SH_event_dict[year] = {}
    MHP_event_dict[year] = {}
    no_hazard_boolean_dict[year] = {}
    SH_boolean_dict[year] = {}
    MHP_boolean_dict[year] = {}
    no_hazard_or_SH_boolean_dict[year] = {}
    SH_or_MHP_boolean_dict[year] = {}
    
    SH_sub = AE[(AE['start_year']==year) | (AE['end_year']==year)].reset_index(drop=True)
    MHP_sub = MHP[(MHP['start_year']==year) | (MHP['end_year']==year)].reset_index(drop=True)

    for state in state_list:
        #print(f'State: {state}')
        # Define nested structure of dictionaries
        SH_count_dict[year][state] = {}
        MHP_count_dict[year][state] = {}
        SH_event_dict[year][state] = {}
        MHP_event_dict[year][state] = {}
        no_hazard_boolean_dict[year][state] = {}
        SH_boolean_dict[year][state] = {}
        MHP_boolean_dict[year][state] = {}
        no_hazard_or_SH_boolean_dict[year][state] = {}
        SH_or_MHP_boolean_dict[year][state] = {}
        
        county_state_list = us_county_polygons.loc[us_county_polygons['STATEFP'] == state, ['GEOID','COUNTYFP','geometry']]

        # Currently using geoid to index, could user countyfp instead, would have to change some things
        for county in county_state_list['GEOID']:
            #print(county)
            
            # SPATIAL GEOMETRY FILTER APPROACH
            ###############################################################
            # county_geom = us_county_polygons.loc[us_county_polygons['GEOID'] == str(county), 'geometry'].iloc[0]
            # #single spatial filter
            # tree = shapely.STRtree(SH_sub.Geometry.values) # Make tree too see Geometry overlap
            # arr1 = np.transpose(tree.query(county_geom, predicate='intersects'))  # Find intersecting hazards with the area of interest
            # # crop hazard data to relevant regions 
            # SH_sub_county = SH_sub.loc[np.sort(arr1)].reset_index(drop=True) # Remove the hazards that do not intersect with the area of interest	
            # # multi spatial filter
            # tree2 = shapely.STRtree(MHP_sub.Geometry.values) # Make tree too see Geometry overlap
            # arr2 = np.transpose(tree2.query(county_geom, predicate='intersects'))  # Find intersecting hazards with the area of interest
            # # crop hazard data to relevant regions 
            # MHP_sub_county = MHP_sub.loc[np.sort(arr2)].reset_index(drop=True) # Remove the hazards that do not intersect with the area of interest	
            ###############################################################
            
            # NON GEOMETRY SPATIAL FILTER APPROACH, FILTER VIA COUNTY GEOID
            ###############################################################
            SH_sub_county = SH_sub[SH_sub['GEOID']==county].reset_index(drop=True) # Remove the hazards that do not intersect with the area of interest	
            MHP_sub_county = MHP_sub[MHP_sub['GEOID']==county].reset_index(drop=True) # Remove the hazards that do not intersect with the area of interest	
            ###############################################################

            
            # Add single to dict
            # Check if there are matching single events in the multi event, this will be true if multi is true, then remove the multi events from the single before adding to dict
            multi_events = set(MHP_sub_county['EVENT_ID'].unique())
            single_events = set(SH_sub_county['EVENT_ID'].unique())
            single_only_events = single_events.symmetric_difference(multi_events) #find the 'code' ids of events in only single
            
            # Remove the single hazards that make up a valid multihazard for that location, so its single hazard only events
            SH_only_sub_county = SH_sub_county[SH_sub_county['EVENT_ID'].isin(single_only_events)]
            
            # If single hazard, add the count of unique single hazard events to the dict
            # Could alternatively use the 'id' as a unique field, I believe both 'code' and 'id' are unique for the single hazards
            if len(SH_only_sub_county)>0:
                SH_boolean_dict[year][state][county] = True
                SH_count_dict[year][state][county] = len(SH_only_sub_county['EVENT_ID'].unique())
                SH_event_dict[year][state][county] = SH_only_sub_county['EVENT_ID'].unique().tolist()
            else:
                SH_boolean_dict[year][state][county] = False
                SH_count_dict[year][state][county] = 0
                SH_event_dict[year][state][county] = []

            # Add multi to dict
            multi_duplicated_events = MHP_sub_county["PAIR_ID"][MHP_sub_county["PAIR_ID"].duplicated(keep=False)]
            multi_filtered_df = MHP_sub_county[MHP_sub_county["PAIR_ID"].isin(multi_duplicated_events)]    
            # If multihazard, add the count of unique multihazard events to dict
            # If len(MHP_sub_county['code'].unique().tolist())>0:
            if len(multi_filtered_df)>0:
                MHP_boolean_dict[year][state][county] = True
                MHP_count_dict[year][state][county] = len(multi_filtered_df['PAIR_ID'].unique())
                MHP_event_dict[year][state][county] = multi_filtered_df['PAIR_ID'].unique().tolist()
                #OR could add the 'code' id of the single hazard events that make up a multihazard
                #MHP_event_dict[year][state][county] = {multi_filtered_df['code'].unique().tolist()}
            else:
                MHP_boolean_dict[year][state][county] = False
                MHP_count_dict[year][state][county] = 0
                MHP_event_dict[year][state][county] = []
            
            # Add no hazard to dict
            if ((len(SH_only_sub_county)==0) & (len(multi_filtered_df)==0)):
                no_hazard_boolean_dict[year][state][county] = True
            else:
                no_hazard_boolean_dict[year][state][county] = False

            # Add no single only hazard to dict, could remove the first two conditions,
            if (((len(SH_only_sub_county)==0) | (len(SH_only_sub_county)>0)) & (len(multi_filtered_df)==0)):
                no_hazard_or_SH_boolean_dict[year][state][county] = True
            else:
                no_hazard_or_SH_boolean_dict[year][state][county] = False

            # Add single hazard or multi-hazard (i.e. inverse of no hazard) to dict
            if ((len(SH_only_sub_county)>0) | (len(multi_filtered_df)>0)):
                SH_or_MHP_boolean_dict[year][state][county] = True
            else:
                SH_or_MHP_boolean_dict[year][state][county] = False


#Save final dictionaries as pickle

with open(Hazard_Dict_Output_Path+f'\\NCEI_County_SH_only_event_dict.pkl', 'wb') as file:
    pickle.dump(SH_event_dict, file)

with open(Hazard_Dict_Output_Path+f'\\NCEI_County_MHP_event_dict.pkl', 'wb') as file:
    pickle.dump(MHP_event_dict, file)

with open(Hazard_Dict_Output_Path+f'\\NCEI_County_MHP_count_dict.pkl', 'wb') as file:
    pickle.dump(MHP_count_dict, file)

with open(Hazard_Dict_Output_Path+f'\\NCEI_County_SH_count_dict.pkl', 'wb') as file:
    pickle.dump(SH_count_dict, file)

with open(Hazard_Dict_Output_Path+f'\\NCEI_County_NH_boolean_dict.pkl', 'wb') as file:
    pickle.dump(no_hazard_boolean_dict, file)
    
with open(Hazard_Dict_Output_Path+f'\\NCEI_County_SH_boolean_dict.pkl', 'wb') as file:
    pickle.dump(SH_boolean_dict, file)

with open(Hazard_Dict_Output_Path+f'\\NCEI_County_MHP_boolean_dict.pkl', 'wb') as file:
    pickle.dump(MHP_boolean_dict, file)

with open(Hazard_Dict_Output_Path+f'\\NCEI_County_SH_NH_boolean_dict.pkl', 'wb') as file:
    pickle.dump(no_hazard_or_SH_boolean_dict, file)

with open(Hazard_Dict_Output_Path+f'\\NCEI_County_SH_MHP_boolean_dict.pkl', 'wb') as file:
    pickle.dump(SH_or_MHP_boolean_dict, file)

print("All hazard dicts saved as pickle")


# Function to access the values from the x3 nested dictionaries
def get_values(data):
    values = []
    for value in data.values():
        if isinstance(value, dict):  # Check if the value is another dictionary
            values.extend(get_values(value))  # Recursively collect values
        else:
            values.extend(value)  # Add the list of values directly
    return values


# # Load single-only hazard event dict
# with open(Hazard_Dict_Output_Path+f'\\NCEI_County_SH_only_event_dict_{start_year}-{end_year}.pkl', 'rb') as file:
#     SH_event_dict = pickle.load(file)

# Subset single hazard events to single-only hazard (single hazards that do not make up a multi-hazard pair)
single_only_hazard_events = get_values(SH_event_dict)
SH = AE[AE['EVENT_ID'].isin(single_only_hazard_events)].reset_index(drop=True)

SH.to_parquet(
    rf"{Hazard_Eventset_Output_Path}/SH_{inj}inj_{dth}dth_{c}c_{p}p_lag{time_lag_int}_{start_year}-{end_year}.parquet.gz",
    compression="gzip",
)

MHP = MHP.reset_index(drop=True)

MHP.to_parquet(
    rf"{Hazard_Eventset_Output_Path}/MHP_{inj}inj_{dth}dth_{c}c_{p}p_lag{time_lag_int}_{start_year}-{end_year}.parquet.gz",
    compression="gzip",
)

MH = MHP.drop_duplicates(subset='EVENT_ID', keep='first', ignore_index=True, inplace=False).reset_index(drop=True)

MH.to_parquet(
    rf"{Hazard_Eventset_Output_Path}/MH_{inj}inj_{dth}dth_{c}c_{p}p_lag{time_lag_int}_{start_year}-{end_year}.parquet.gz",
    compression="gzip",
)

event_set_count_outputs = [
    f"Total Number of Hazard Events: {len(AE)}",
    f"Number of Single-Hazard Only Events (SH): {len(SH)}",
    f"Number of Multi-Hazard Pairs (MHP): {int(len(MHP)/2)}",
    f"Number of Unique Multi-Hazard Events (MH): {len(MH)}",
]

event_set_count_output_txt_path = rf"{Hazard_Eventset_Output_Path}/eventset_counts_{inj}inj_{dth}dth_{c}c_{p}p_lag{time_lag_int}_{start_year}-{end_year}.txt"

with open(event_set_count_output_txt_path, "w", encoding="utf-8") as fp:
    for line in event_set_count_outputs:
        fp.write(line + "\n")

print(f"Counts saved to: {event_set_count_output_txt_path}")


print(f'Total Number of Hazard Events (AE): {len(AE)}')
print(f'Number of Single-Hazard Only Events (SH):{len(SH)}')
print(f'Number of Multi-Hazard Pairs (MHP):{int(len(MHP)/2)}')
print(f'Number of Unique Multi-Hazard Events (MH):{len(MH.drop_duplicates(subset='EVENT_ID'))}')

