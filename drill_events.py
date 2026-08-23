import re
import unicodedata

import pandas as pd
from geopy.distance import geodesic

WELLS_FILE = "datasets/wells.csv"
EVENTS_FILE = "datasets/drilling_events.csv"


def normalize_well_id(value):
    """Return a comparison-only canonical well ID; never replace CSV IDs."""
    if pd.isna(value):
        return ""
    # Inspection shows CSV IDs such as W001 while the requested ID is W-001.
    # This controlled normalization reconciles cosmetic formatting only.
    normalized = unicodedata.normalize("NFKC", str(value))
    normalized = "".join(
        char for char in normalized
        if not char.isspace() and unicodedata.category(char) != "Cf"
    )
    normalized = normalized.strip().upper()
    return re.sub(r"[-_\u2010-\u2015]", "", normalized)


print("Loading datasets...")
wells = pd.read_csv(WELLS_FILE)
events = pd.read_csv(EVENTS_FILE)
print(f"Wells loaded: {len(wells)}")
print(f"Events loaded: {len(events)}")
print("\nWells columns:")
print(wells.columns.tolist())
print("\nEvents columns:")
print(events.columns.tolist())

# Original ID columns are retained as the dataset keys. Helpers are matching only.
wells["_well_id_normalized"] = wells["well_id"].apply(normalize_well_id)
events["_well_id_normalized"] = events["well_id"].apply(normalize_well_id)


def print_id_debugging():
    """Print raw diagnostics before matching, including hidden characters via repr."""
    print("\n==========================================")
    print("WELL ID DEBUGGING")
    print("==========================================")
    print("\nFirst 20 wells['well_id'] values:")
    print(wells["well_id"].head(20).to_string(index=False))
    print("\nFirst 20 events['well_id'] values:")
    print(events["well_id"].head(20).to_string(index=False))
    print("\nFirst 20 wells['well_id'] repr values:")
    print(wells["well_id"].head(20).map(repr).to_string(index=False))
    print("\nFirst 20 events['well_id'] repr values:")
    print(events["well_id"].head(20).map(repr).to_string(index=False))
    print(f"\nwells['well_id'] dtype: {wells['well_id'].dtype}")
    print(f"events['well_id'] dtype: {events['well_id'].dtype}")
    print("\nUnique well IDs containing '001':")
    print(wells.loc[wells["well_id"].astype(str).str.contains("001", na=False), "well_id"]
          .drop_duplicates().map(repr).to_string(index=False))
    print("\nUnique event IDs containing '001':")
    print(events.loc[events["well_id"].astype(str).str.contains("001", na=False), "well_id"]
          .drop_duplicates().map(repr).to_string(index=False))
    print(f"\nDoes 'W-001' exist exactly in wells? {'W-001' in wells['well_id'].values}")
    print("Does 'W-001' exist after normalization? "
          f"{normalize_well_id('W-001') in wells['_well_id_normalized'].values}")


print_id_debugging()

# Keep the original coordinate-cleaning behavior.
wells["latitude"] = pd.to_numeric(wells["latitude"], errors="coerce")
wells["longitude"] = pd.to_numeric(wells["longitude"], errors="coerce")
wells = wells.dropna(subset=["latitude", "longitude"]).copy()


def matching_wells(well_id):
    """Return all rows whose normalized ID matches the supplied ID."""
    return wells.loc[
        wells["_well_id_normalized"] == normalize_well_id(well_id)
    ].copy()


def find_well(well_id):
    """Return the first matching wells.csv row, or None if it does not exist."""
    matches = matching_wells(well_id)
    return None if matches.empty else matches.iloc[0]


def print_available_well_ids(limit=20):
    available = wells["well_id"].drop_duplicates().astype(str).tolist()
    print(f"Available well IDs (first {min(limit, len(available))}):")
    print(", ".join(available[:limit]))
    if len(available) > limit:
        print(f"... ({len(available)} total unique IDs)")


def find_nearby_wells(selected_well_id, radius_km=10):
    """Find offset wells within radius_km using normalized IDs for selection."""
    selected = find_well(selected_well_id)
    if selected is None:
        print(f"\nERROR: Well {selected_well_id!r} was not found after normalization.")
        print_available_well_ids()
        return pd.DataFrame()

    selected_location = (selected["latitude"], selected["longitude"])
    selected_normalized_id = selected["_well_id_normalized"]
    nearby_wells = []
    for _, well in wells.iterrows():
        if well["_well_id_normalized"] == selected_normalized_id:
            continue
        distance_km = geodesic(
            selected_location, (well["latitude"], well["longitude"])
        ).km
        if distance_km <= radius_km:
            nearby_wells.append({
                "well_id": well["well_id"],
                "_well_id_normalized": well["_well_id_normalized"],
                "well_name": well["well_name"], "field": well["field"],
                "latitude": well["latitude"], "longitude": well["longitude"],
                "distance_km": round(distance_km, 2),
            })
    result = pd.DataFrame(nearby_wells)
    return result.sort_values("distance_km").reset_index(drop=True) if not result.empty else result


def get_historical_events(nearby_wells):
    """Return events matched to nearby wells through normalized helper IDs."""
    if nearby_wells.empty:
        return pd.DataFrame()
    nearby_ids = nearby_wells["_well_id_normalized"].tolist()
    historical_events = events.loc[
        events["_well_id_normalized"].isin(nearby_ids)
    ].copy()
    historical_events = historical_events.merge(
        nearby_wells[["_well_id_normalized", "well_name", "distance_km"]],
        on="_well_id_normalized", how="left",
    )
    return historical_events.sort_values(
        ["distance_km", "depth_m"]
    ).reset_index(drop=True)


CURRENT_WELL = "W-001"
RADIUS_KM = 10

print("\n==========================================")
print("CURRENT WELL")
print("==========================================")
print(f"Selected well: {CURRENT_WELL}")
print(f"Search radius: {RADIUS_KM} km")
current_matches = matching_wells(CURRENT_WELL)
print(f"Number of matched current wells: {len(current_matches)}")

test_well = CURRENT_WELL
if current_matches.empty:
    # Do not fabricate W-001; only demonstrate with a valid dataset ID.
    test_well = str(wells.iloc[0]["well_id"])
    print(f"W-001 is not present. Testing with valid CSV ID: {test_well}")
    print_available_well_ids()

matched_well = find_well(test_well)
if matched_well is not None:
    print(f"Matched CSV ID: {matched_well['well_id']}")

nearby = find_nearby_wells(test_well, RADIUS_KM)
print(f"Number of nearby wells: {len(nearby)}")
print("\n==========================================")
print("NEARBY WELLS")
print("==========================================")
if nearby.empty:
    print("No nearby wells found.")
else:
    print(nearby[["well_id", "well_name", "field", "distance_km"]].to_string(index=False))

historical_events = get_historical_events(nearby)
print(f"Number of historical events found: {len(historical_events)}")
print("\n==========================================")
print("HISTORICAL DRILLING EVENTS")
print("==========================================")
if historical_events.empty:
    print("No historical events found for nearby wells.")
else:
    print(historical_events[[
        "well_id", "well_name", "distance_km", "depth_m", "formation", "event_type",
        "severity", "event_description", "duration_hours", "operational_impact",
    ]].to_string(index=False))
