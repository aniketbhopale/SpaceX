import pandas as pd
import folium
from folium.plugins import MarkerCluster, Search
from geopy.distance import geodesic
from branca.element import Element

CSV_FILE = "datasets/wells.csv"
OUTPUT_FILE = "oil_well_map.html"

DEFAULT_ZOOM = 10
DEFAULT_RADIUS_KM = 10

print("Loading wells.csv...")

wells = pd.read_csv(CSV_FILE)

print(f"Total rows in CSV: {len(wells)}")

print("\nColumns:")
print(wells.columns.tolist())



required_columns = [
    "well_id",
    "well_name",
    "latitude",
    "longitude",
    "field",
    "total_depth_m",
    "primary_formation",
    "well_status",
    "spud_date",
    "well_role"
]

missing_columns = [
    col for col in required_columns
    if col not in wells.columns
]

if missing_columns:
    raise ValueError(
        f"Missing columns in wells.csv: {missing_columns}"
    )

wells["latitude"] = pd.to_numeric(
    wells["latitude"],
    errors="coerce"
)

wells["longitude"] = pd.to_numeric(
    wells["longitude"],
    errors="coerce"
)



print("\nCoordinate information:")
print(
    wells[
        ["well_id", "latitude", "longitude"]
    ].head(10)
)

print("\nMissing coordinates:")
print(
    wells[
        ["latitude", "longitude"]
    ].isna().sum()
)



wells = wells.dropna(
    subset=[
        "latitude",
        "longitude"
    ]
).copy()


print(
    f"\nWells remaining after coordinate cleaning: "
    f"{len(wells)}"
)



print("\nLatitude range:")
print(
    wells["latitude"].min(),
    "to",
    wells["latitude"].max()
)

print("\nLongitude range:")
print(
    wells["longitude"].min(),
    "to",
    wells["longitude"].max()
)



center_lat = wells["latitude"].mean()
center_lon = wells["longitude"].mean()

print(
    f"\nMap center: "
    f"{center_lat:.6f}, "
    f"{center_lon:.6f}"
)

m = folium.Map(

    location=[
        center_lat,
        center_lon
    ],

    zoom_start=DEFAULT_ZOOM,

    tiles="CartoDB positron",

    control_scale=True
)

def get_status_color(status):

    status = str(status).strip().lower()

    colors = {

        "active": "green",

        "completed": "blue",

        "drilled": "orange",

        "abandoned": "red",

        "suspended": "purple",

        "plugged": "darkred"
    }

    return colors.get(
        status,
        "gray"
    )



wells_layer = folium.FeatureGroup(
    name="Oil Wells",
    show=True
)

wells_layer.add_to(m)



marker_cluster = MarkerCluster(
    name="Well Clusters"
)

marker_cluster.add_to(wells_layer)


print("\nAdding wells to map...")

for _, well in wells.iterrows():

    well_id = str(well["well_id"])

    well_name = str(
        well["well_name"]
    )

    latitude = float(
        well["latitude"]
    )

    longitude = float(
        well["longitude"]
    )

    field = str(
        well["field"]
    )

    depth = str(
        well["total_depth_m"]
    )

    formation = str(
        well["primary_formation"]
    )

    status = str(
        well["well_status"]
    )

    role = str(
        well["well_role"]
    )

    spud_date = str(
        well["spud_date"]
    )

    color = get_status_color(
        status
    )



    popup_html = f"""

    <div style="
        width: 280px;
        font-family: Arial, sans-serif;
    ">

        <div style="
            background: #1f2937;
            color: white;
            padding: 12px;
            border-radius: 8px 8px 0 0;
        ">

            <div style="
                font-size: 18px;
                font-weight: bold;
            ">
                {well_name}
            </div>

            <div style="
                font-size: 12px;
                margin-top: 4px;
                color: #d1d5db;
            ">
                {well_id}
            </div>

        </div>


        <div style="
            padding: 10px;
        ">

            <b>Field:</b>
            {field}
            <br><br>

            <b>Total Depth:</b>
            {depth} m
            <br><br>

            <b>Formation:</b>
            {formation}
            <br><br>

            <b>Status:</b>

            <span style="
                color: {color};
                font-weight: bold;
            ">
                {status}
            </span>

            <br><br>

            <b>Well Role:</b>
            {role}
            <br><br>

            <b>Spud Date:</b>
            {spud_date}
            <br><br>

            <b>Coordinates:</b>
            {latitude:.5f},
            {longitude:.5f}

        </div>

    </div>

    """


    popup = folium.Popup(
        popup_html,
        max_width=350
    )



    tooltip = folium.Tooltip(
        f"""
        <b>{well_id}</b>
        <br>
        {well_name}
        <br>
        <span style="color:{color};">
            ● {status}
        </span>
        """
    )


    marker = folium.CircleMarker(

        location=[
            latitude,
            longitude
        ],

        radius=9,

        color="white",

        weight=2,

        fill=True,

        fill_color=color,

        fill_opacity=0.9,

        popup=popup,

        tooltip=tooltip
    )


    marker.add_to(
        marker_cluster
    )


print(
    f"Successfully added "
    f"{len(wells)} wells."
)



# Convert wells to GeoJSON for accurate property search
geojson_features = []
for _, well in wells.iterrows():
    w_id = str(well["well_id"]).strip()
    w_name = str(well["well_name"]).strip()
    geojson_features.append({
        "type": "Feature",
        "geometry": {
            "type": "Point",
            "coordinates": [float(well["longitude"]), float(well["latitude"])],
        },
        "properties": {
            "well_id": w_id,
            "search_label": f"{w_id} - {w_name}",
        },
    })

search_layer = folium.GeoJson(
    {"type": "FeatureCollection", "features": geojson_features},
    name="Search Wells",
    style_function=lambda x: {"opacity": 0, "fillOpacity": 0},
    tooltip=folium.GeoJsonTooltip(fields=["search_label"], aliases=["Well:"]),
    show=True
).add_to(m)

Search(
    layer=search_layer,
    geom_type="Point",
    search_label="well_id",
    search_zoom=14,
    placeholder="Search Well ID (e.g., W001)...",
    collapsed=False,
    position="topright"
).add_to(m)



legend_html = """

<div style="
    position: fixed;

    bottom: 30px;
    left: 30px;

    width: 180px;

    background: white;

    padding: 15px;

    border-radius: 10px;

    z-index: 9999;

    font-family: Arial;

    box-shadow:
        0 3px 12px rgba(0,0,0,0.2);
">

    <b style="
        font-size: 15px;
    ">
        WELL STATUS
    </b>

    <hr>

    <div>
        <span style="
            color: green;
            font-size: 20px;
        ">●</span>
        Active
    </div>

    <div>
        <span style="
            color: blue;
            font-size: 20px;
        ">●</span>
        Completed
    </div>

    <div>
        <span style="
            color: orange;
            font-size: 20px;
        ">●</span>
        Drilled
    </div>

    <div>
        <span style="
            color: red;
            font-size: 20px;
        ">●</span>
        Abandoned
    </div>

    <div>
        <span style="
            color: gray;
            font-size: 20px;
        ">●</span>
        Other

    </div>

</div>

"""

m.get_root().html.add_child(
    Element(legend_html)
)


title_html = """

<div style="
    position: fixed;

    top: 15px;

    left: 50%;

    transform: translateX(-50%);

    z-index: 9999;

    background: rgba(255,255,255,0.95);

    padding: 12px 25px;

    border-radius: 10px;

    box-shadow:
        0 3px 12px rgba(0,0,0,0.15);

    font-family: Arial;

    text-align: center;
">

    <div style="
        font-size: 18px;
        font-weight: bold;
    ">
        🛢️ Oil & Gas Well Monitoring Map
    </div>

    <div style="
        font-size: 11px;
        color: #6b7280;
    ">
        Interactive Offset Well Visualization
    </div>

</div>

"""

m.get_root().html.add_child(
    Element(title_html)
)


if len(wells) > 0:

    bounds = [

        [
            wells["latitude"].min(),
            wells["longitude"].min()
        ],

        [
            wells["latitude"].max(),
            wells["longitude"].max()
        ]

    ]

    m.fit_bounds(
        bounds,
        padding=(30, 30)
    )

else:

    print(
        "ERROR: No wells available "
        "after coordinate cleaning."
    )


folium.LayerControl(
    collapsed=False,
    position="topright"
).add_to(m)



def find_nearby_wells(
    well_id,
    radius_km=DEFAULT_RADIUS_KM
):

    selected = wells[
        wells["well_id"].astype(str)
        == str(well_id)
    ]

    if selected.empty:

        print(
            f"Well {well_id} not found."
        )

        return pd.DataFrame()


    selected = selected.iloc[0]


    selected_location = (

        selected["latitude"],

        selected["longitude"]

    )


    results = []


    for _, well in wells.iterrows():

        if str(
            well["well_id"]
        ) == str(well_id):

            continue


        other_location = (

            well["latitude"],

            well["longitude"]

        )


        distance = geodesic(

            selected_location,

            other_location

        ).km


        if distance <= radius_km:

            results.append({

                "well_id":
                    well["well_id"],

                "well_name":
                    well["well_name"],

                "field":
                    well["field"],

                "latitude":
                    well["latitude"],

                "longitude":
                    well["longitude"],

                "distance_km":
                    round(distance, 2)

            })


    result = pd.DataFrame(
        results
    )


    if not result.empty:

        result = result.sort_values(
            "distance_km"
        )


    return result



m.save(
    OUTPUT_FILE
)



print("MAP CREATED SUCCESSFULLY ....!!!")

print(
    f"Output file: {OUTPUT_FILE}"
)

print(
    f"Wells plotted: {len(wells)}"
)

print(
    f"Center: "
    f"{center_lat:.6f}, "
    f"{center_lon:.6f}"
)

print(
    "\nOpen this file in Chrome:"
)

print(
    OUTPUT_FILE
)