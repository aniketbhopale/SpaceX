
import os
import re
import unicodedata
import json
import altair as alt
import folium
from folium.plugins import Search
from geopy.distance import geodesic
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import streamlit as st
import streamlit.components.v1 as components
from dotenv import load_dotenv
from openai import OpenAI

# Reads OPENAI_API_KEY from .env without displaying it in the dashboard or logs.
load_dotenv()
CHAT_MODEL = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")

# Section 1: Page setup and UI styling
st.set_page_config(
    page_title="NWIS — Offset Well Intelligence Platform",
    page_icon="🛢️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for the clean light-industrial dashboard look
st.markdown(
    """
    <style>
    /* Main application background and default text styling */
    .stApp {
        background-color: #f8fafc;
        color: #0f172a;
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
    }
    
    /* Summary cards at the top of the dashboard */
    .nwis-card {
        background-color: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 14px 18px;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
        margin-bottom: 12px;
    }
    
    /* Section headers */
    .nwis-header {
        font-size: 1.1rem;
        font-weight: 700;
        color: #1e293b;
        border-bottom: 2px solid #e2e8f0;
        padding-bottom: 8px;
        margin-bottom: 12px;
        letter-spacing: 0.3px;
    }
    
    /* Dynamic risk alert banners with colored accent borders */
    .risk-banner-critical {
        background: #fef2f2;
        border-left: 6px solid #dc2626;
        border-top: 1px solid #fee2e2;
        border-right: 1px solid #fee2e2;
        border-bottom: 1px solid #fee2e2;
        padding: 16px 20px;
        border-radius: 6px;
        margin-bottom: 14px;
        box-shadow: 0 1px 3px rgba(220, 38, 38, 0.1);
    }
    
    .risk-banner-high {
        background: #fff7ed;
        border-left: 6px solid #ea580c;
        border-top: 1px solid #ffedd5;
        border-right: 1px solid #ffedd5;
        border-bottom: 1px solid #ffedd5;
        padding: 16px 20px;
        border-radius: 6px;
        margin-bottom: 14px;
        box-shadow: 0 1px 3px rgba(234, 88, 12, 0.1);
    }
    
    .risk-banner-caution {
        background: #fefce8;
        border-left: 6px solid #ca8a04;
        border-top: 1px solid #fef9c3;
        border-right: 1px solid #fef9c3;
        border-bottom: 1px solid #fef9c3;
        padding: 16px 20px;
        border-radius: 6px;
        margin-bottom: 14px;
        box-shadow: 0 1px 3px rgba(202, 138, 4, 0.1);
    }
    
    .risk-banner-low {
        background: #f0fdf4;
        border-left: 6px solid #16a34a;
        border-top: 1px solid #dcfce7;
        border-right: 1px solid #dcfce7;
        border-bottom: 1px solid #dcfce7;
        padding: 16px 20px;
        border-radius: 6px;
        margin-bottom: 14px;
        box-shadow: 0 1px 3px rgba(22, 163, 74, 0.1);
    }
    
    .metric-title {
        color: #64748b;
        font-size: 0.78rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .metric-val {
        font-family: -apple-system, BlinkMacSystemFont, monospace;
        font-size: 1.15rem;
        font-weight: 700;
        color: #0f172a;
        margin-top: 4px;
    }
    
    /* Scrollable container for the list of historical offset events */
    .scrollable-events-container {
        max-height: 420px;
        overflow-y: auto;
        padding-right: 6px;
    }
    </style>
""",
    unsafe_allow_html=True,
)


# Section 2: Load datasets and helper functions
def normalize_well_id(value):
    """Cleans well IDs by removing spaces and dashes for consistent matching."""
    if pd.isna(value):
        return ""
    normalized = unicodedata.normalize("NFKC", str(value))
    normalized = "".join(
        char
        for char in normalized
        if not char.isspace() and unicodedata.category(char) != "Cf"
    )
    normalized = normalized.strip().upper()
    return re.sub(r"[-_\u2010-\u2015]", "", normalized)


def resolve_dataset_path(filename):
    """Automatically finds the CSV file in root or in the datasets/ folder."""
    candidates = [
        os.path.join("datasets", filename),
        filename,
        os.path.join("..", "datasets", filename),
        os.path.join("..", filename),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    return filename


def resolve_report_path(well_id, report_type="DDR"):
    """Return the actual local DDR/WCR PDF for a well, preferring digital reports."""
    report_name = f"{well_id}_{report_type}.pdf"
    candidates = [
        os.path.join("data", "reports", report_name),
        os.path.join("data", "scanned_reports", report_name),
        os.path.join("data", "scanned_reports", f"{well_id}_{report_type}_scanned.pdf"),
    ]
    for path in candidates:
        if os.path.isfile(path):
            return path
    return None


@st.cache_data(show_spinner=False)
def load_report_file(report_path):
    """Cache local PDF bytes for Streamlit's reliable file-download control."""
    with open(report_path, "rb") as report_file:
        return report_file.read()


def _json_value(value):
    """Convert Pandas/NumPy values to compact JSON-safe chatbot context values."""
    if pd.isna(value):
        return None
    if isinstance(value, np.generic):
        return value.item()
    return value


def build_well_chat_context(
    selected_meta, nearby_wells, nearby_events, formation, depth, telemetry, risk
):
    """Build the bounded evidence context used by the well-information chatbot."""
    well_info = {
        key: _json_value(value)
        for key, value in selected_meta.drop(labels=["_id_norm"], errors="ignore").items()
    }
    nearby_columns = [
        "well_id", "well_name", "field", "distance_km", "total_depth_m",
        "primary_formation", "well_status",
    ]
    event_columns = [
        "well_id", "distance_km", "depth_m", "formation", "event_type",
        "severity", "duration_hours", "operational_impact",
    ]
    nearby_summary = (
        nearby_wells.reindex(columns=nearby_columns).head(30).map(_json_value).to_dict("records")
        if not nearby_wells.empty else []
    )
    event_summary = (
        nearby_events.reindex(columns=event_columns).head(80).map(_json_value).to_dict("records")
        if not nearby_events.empty else []
    )
    return {
        "current_well": well_info,
        "operational_context": {
            "bit_depth_m": depth,
            "formation_at_depth": formation,
            "telemetry": telemetry,
            "risk_assessment": {
                key: _json_value(value)
                for key, value in risk.items()
                if key not in {"similar_events_df", "badge_style", "level_display"}
            },
        },
        "nearby_wells": nearby_summary,
        "historical_offset_events": event_summary,
    }


def ask_well_chatbot(question, context, chat_history):
    """Answer from supplied well evidence only; the OpenAI key is never exposed."""
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is missing. Add it to .env and restart Streamlit.")

    system_prompt = """
You are the NWIS Well Information Assistant for an oil and gas offset-well
intelligence dashboard. Answer using ONLY the supplied structured evidence.
Do not invent well properties, depths, locations, formations, events, or risk
facts. If evidence is missing, state that it is not available in the loaded
datasets. Be concise, identify the well/event evidence behind claims, and make
clear that this dashboard aid does not replace approved drilling procedures or
real-time operational controls.
""".strip()
    prior_messages = [
        {"role": message["role"], "content": message["content"]}
        for message in chat_history[-6:]
    ]
    user_message = (
        "CURRENT DATA EVIDENCE:\n"
        + json.dumps(context, default=str, ensure_ascii=False)
        + "\n\nUSER QUESTION:\n"
        + question
    )
    client = OpenAI()
    response = client.responses.create(
        model=CHAT_MODEL,
        instructions=system_prompt,
        input=prior_messages + [{"role": "user", "content": user_message}],
        store=False,
    )
    return response.output_text.strip() or "No answer was returned by the model."


@st.cache_data
def load_all_datasets():
    """Loads and caches all 5 required CSV files."""
    wells_df = pd.read_csv(resolve_dataset_path("wells.csv"))
    events_df = pd.read_csv(resolve_dataset_path("drilling_events.csv"))
    params_df = pd.read_csv(resolve_dataset_path("drilling_parameters.csv"))
    formations_df = pd.read_csv(resolve_dataset_path("formations.csv"))
    well_formations_df = pd.read_csv(
        resolve_dataset_path("well_formations.csv")
    )

    # Clean up coordinates to ensure numbers only
    wells_df["latitude"] = pd.to_numeric(wells_df["latitude"], errors="coerce")
    wells_df["longitude"] = pd.to_numeric(
        wells_df["longitude"], errors="coerce"
    )
    wells_df = wells_df.dropna(subset=["latitude", "longitude"]).copy()

    # Create temporary normalized columns for fast, error-free joins
    wells_df["_id_norm"] = wells_df["well_id"].apply(normalize_well_id)
    events_df["_id_norm"] = events_df["well_id"].apply(normalize_well_id)
    params_df["_id_norm"] = params_df["well_id"].apply(normalize_well_id)
    well_formations_df["_id_norm"] = well_formations_df["well_id"].apply(
        normalize_well_id
    )

    return wells_df, events_df, params_df, formations_df, well_formations_df


# Load all tables
wells_df, events_df, params_df, formations_df, well_formations_df = (
    load_all_datasets()
)


# Section 3: Stratigraphy and distance calculations
def get_formation_at_depth(well_id_norm, depth):
    """Finds which rock layer (Sandstone, Shale, etc.) exists at a given depth."""
    sub = well_formations_df[well_formations_df["_id_norm"] == well_id_norm]
    if sub.empty:
        return "Unknown Formation"

    match = sub[(sub["top_depth_m"] <= depth) & (sub["bottom_depth_m"] >= depth)]
    if not match.empty:
        return match.iloc[0]["formation_name"]

    # Fallback to the closest boundary if on edge
    if depth < sub["top_depth_m"].min():
        return sub.sort_values("top_depth_m").iloc[0]["formation_name"]
    return sub.sort_values("bottom_depth_m", ascending=False).iloc[0][
        "formation_name"
    ]


def get_nearby_wells(selected_norm_id, radius_km=10.0):
    """Calculates geographic distance (km) to find all offset wells in the corridor."""
    target_row = wells_df[wells_df["_id_norm"] == selected_norm_id]
    if target_row.empty:
        return pd.DataFrame()

    target_loc = (
        target_row.iloc[0]["latitude"],
        target_row.iloc[0]["longitude"],
    )
    results = []

    for _, row in wells_df.iterrows():
        # Exclude the active well itself
        if row["_id_norm"] == selected_norm_id:
            continue

        dist = geodesic(target_loc, (row["latitude"], row["longitude"])).km
        if dist <= radius_km:
            results.append(
                {
                    "well_id": row["well_id"],
                    "_id_norm": row["_id_norm"],
                    "well_name": row["well_name"],
                    "field": row["field"],
                    "latitude": row["latitude"],
                    "longitude": row["longitude"],
                    "total_depth_m": row["total_depth_m"],
                    "primary_formation": row["primary_formation"],
                    "well_status": row["well_status"],
                    "distance_km": round(dist, 2),
                }
            )

    res_df = pd.DataFrame(results)
    if not res_df.empty:
        res_df = res_df.sort_values("distance_km").reset_index(drop=True)
    return res_df


# Section 4: Machine learning model for risk prediction
@st.cache_resource
def train_risk_classifier():
    """Builds and trains the Random Forest classifier on historical drilling parameters."""
    df_records = params_df.copy()

    # Step A: Associate each telemetry point with its rock formation
    formations_list = []
    for _, row in df_records.iterrows():
        formations_list.append(
            get_formation_at_depth(row["_id_norm"], row["depth_m"])
        )
    df_records["formation"] = formations_list

    # Step B: Match drilling incidents occurring within +/- 50m of that depth
    labels = []
    severities = []
    for _, row in df_records.iterrows():
        well_events = events_df[
            (events_df["_id_norm"] == row["_id_norm"])
            & (abs(events_df["depth_m"] - row["depth_m"]) <= 50)
        ]
        if not well_events.empty:
            top_event = well_events.iloc[0]
            labels.append(top_event["event_type"])
            severities.append(top_event["severity"])
        else:
            labels.append("Normal")
            severities.append("None")

    df_records["event_label"] = labels
    df_records["severity_label"] = severities

    # Step C: Encode categories into numeric values for the ML model
    le_form = LabelEncoder()
    df_records["formation_encoded"] = le_form.fit_transform(
        df_records["formation"]
    )

    le_event = LabelEncoder()
    df_records["event_encoded"] = le_event.fit_transform(
        df_records["event_label"]
    )

    features = [
        "depth_m",
        "mud_weight_ppg",
        "rop_m_per_hr",
        "rpm",
        "wob_klbf",
        "formation_encoded",
    ]
    X = df_records[features].fillna(0)
    y = df_records["event_encoded"]

    # Step D: Train the model
    model = RandomForestClassifier(
        n_estimators=120, max_depth=12, random_state=42, class_weight="balanced"
    )
    model.fit(X, y)

    return model, le_form, le_event, features


ml_model, le_form, le_event, feature_cols = train_risk_classifier()


def predict_drilling_risk(
    depth, mud_weight, rop, rpm, wob, formation_name, offset_events
):
    """Combines ML probabilities with offset well history to produce an alert."""
    if formation_name in le_form.classes_:
        f_enc = le_form.transform([formation_name])[0]
    else:
        f_enc = 0

    x_input = pd.DataFrame(
        [
            {
                "depth_m": depth,
                "mud_weight_ppg": mud_weight,
                "rop_m_per_hr": rop,
                "rpm": rpm,
                "wob_klbf": wob,
                "formation_encoded": f_enc,
            }
        ]
    )

    # Get ML probability for each hazard type
    probs = ml_model.predict_proba(x_input)[0]
    classes = le_event.classes_
    risk_dict = {classes[i]: probs[i] for i in range(len(classes))}

    hazard_probs = {k: v for k, v in risk_dict.items() if k != "Normal"}
    top_hazard = (
        max(hazard_probs, key=hazard_probs.get) if hazard_probs else "Mud Loss"
    )
    top_hazard_score = hazard_probs.get(top_hazard, 0.0)

    # Check for historical offset incidents near this depth (+/- 150m) or same formation
    similar_depth_events = offset_events[
        (abs(offset_events["depth_m"] - depth) <= 150)
        | (
            offset_events["formation"].str.lower()
            == str(formation_name).lower()
        )
    ]

    has_critical_offset = False
    has_high_offset = False
    matching_wells_count = 0
    min_event_depth, max_event_depth = depth, depth

    if not similar_depth_events.empty:
        matching_wells_count = similar_depth_events["well_id"].nunique()
        min_event_depth = int(similar_depth_events["depth_m"].min())
        max_event_depth = int(similar_depth_events["depth_m"].max())
        has_critical_offset = (
            similar_depth_events["severity"].str.lower() == "critical"
        ).any()
        has_high_offset = (
            similar_depth_events["severity"].str.lower() == "high"
        ).any()

        # Prioritize the most frequent hazard observed in neighboring wells
        offset_top_hazard = similar_depth_events["event_type"].mode()
        if not offset_top_hazard.empty:
            top_hazard = offset_top_hazard[0]

    # Assign risk severity tier based on offset evidence and ML confidence
    if has_critical_offset or top_hazard_score > 0.65:
        level = "CATASTROPHIC"
        level_display = "🚨 CATASTROPHIC RISK"
        badge_style = "risk-banner-critical"
    elif has_high_offset or matching_wells_count >= 2 or top_hazard_score > 0.35:
        level = "HIGH"
        level_display = "🚨 HIGH RISK"
        badge_style = "risk-banner-high"
    elif matching_wells_count == 1 or top_hazard_score > 0.20:
        level = "CAUTIONARY"
        level_display = "⚠️ CAUTIONARY RISK"
        badge_style = "risk-banner-caution"
    else:
        level = "LOW"
        level_display = "✅ LOW RISK"
        badge_style = "risk-banner-low"

    return {
        "level": level,
        "level_display": level_display,
        "badge_style": badge_style,
        "top_hazard": top_hazard if top_hazard != "Normal" else "Mud Loss",
        "confidence": round(top_hazard_score * 100, 1),
        "matching_wells_count": matching_wells_count,
        "depth_range": (min_event_depth, max_event_depth),
        "similar_events_df": similar_depth_events,
    }


# Section 5: Sidebar controls and well selection
st.sidebar.markdown(
    """
    <div style="text-align: center; padding: 4px 0 14px 0;">
        <h2 style="margin: 0; color: #0284c7;">🛢️ eRTMAC-NWIS</h2>
        <p style="margin: 0; font-size: 0.8rem; color: #64748b;">Nearby Wells Intelligence System</p>
    </div>
""",
    unsafe_allow_html=True,
)

st.sidebar.markdown("### 🔍 Select Well")
well_options = wells_df["well_id"].tolist()
default_index = well_options.index("W141") if "W141" in well_options else 0

selected_well_id = st.sidebar.selectbox(
    "Well ID",
    options=well_options,
    index=default_index,
    help="Select a well to view offset risk intelligence.",
)

selected_norm_id = normalize_well_id(selected_well_id)
selected_well_meta = wells_df[wells_df["_id_norm"] == selected_norm_id].iloc[0]

# Search corridor radius slider
radius_km = st.sidebar.slider(
    "Offset Search Radius (km)", min_value=3, max_value=30, value=10, step=1
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 📏 Operational Depth")

max_possible_depth = int(selected_well_meta["total_depth_m"])
default_depth = min(2790, max_possible_depth)

current_depth = st.sidebar.slider(
    "Bit Depth (m)",
    min_value=25,
    max_value=max_possible_depth,
    value=default_depth,
    step=10,
)

# Fetch latest telemetry from parameters table
matched_param = params_df[
    (params_df["_id_norm"] == selected_norm_id)
    & (params_df["depth_m"] <= current_depth)
]

if not matched_param.empty:
    default_mw = float(matched_param.iloc[-1]["mud_weight_ppg"])
    default_rop = float(matched_param.iloc[-1]["rop_m_per_hr"])
    default_rpm = float(matched_param.iloc[-1]["rpm"])
    default_wob = float(matched_param.iloc[-1]["wob_klbf"])
else:
    default_mw, default_rop, default_rpm, default_wob = (
        12.6,
        17.4,
        105.0,
        25.7,
    )

with st.sidebar.expander("Telemetry Adjustments", expanded=False):
    live_mw = st.number_input(
        "Mud Weight (ppg)", value=default_mw, step=0.1, format="%.2f"
    )
    live_rop = st.number_input(
        "ROP (m/hr)", value=default_rop, step=0.5, format="%.1f"
    )
    live_rpm = st.number_input("RPM", value=default_rpm, step=5.0)
    live_wob = st.number_input(
        "Weight on Bit (klbf)", value=default_wob, step=1.0
    )

# Rock formation encountered at the selected depth
current_formation = get_formation_at_depth(selected_norm_id, current_depth)


# Section 6: Identify nearby suspected wells
nearby_df = get_nearby_wells(selected_norm_id, radius_km)

# Retrieve all incidents recorded on nearby offset wells
if not nearby_df.empty:
    nearby_norm_ids = nearby_df["_id_norm"].tolist()
    offset_events = events_df[
        events_df["_id_norm"].isin(nearby_norm_ids)
    ].copy()
    offset_events = offset_events.merge(
        nearby_df[["_id_norm", "distance_km", "well_name"]],
        on="_id_norm",
        how="left",
    )
else:
    offset_events = pd.DataFrame()

# Run the AI risk model
ai_risk = predict_drilling_risk(
    depth=current_depth,
    mud_weight=live_mw,
    rop=live_rop,
    rpm=live_rpm,
    wob=live_wob,
    formation_name=current_formation,
    offset_events=offset_events,
)

# Identify which wells are "suspected" (encountered problems near current depth/formation)
suspected_well_ids = set()
if not offset_events.empty:
    suspected_matches = offset_events[
        (
            offset_events["formation"].str.lower()
            == str(current_formation).lower()
        )
        | (abs(offset_events["depth_m"] - current_depth) <= 120)
    ]
    suspected_well_ids = set(suspected_matches["well_id"].tolist())


# Section 7: Generate interactive Folium map
def create_nwis_map(
    target_meta,
    nearby_wells_df,
    suspected_ids,
    current_depth_val,
    curr_formation_val,
):
    """Draws the light-themed map with the active well, suspected wells, and normal wells."""
    center_lat = target_meta["latitude"]
    center_lon = target_meta["longitude"]

    # Initialize map with clean CartoDB Positron light tiles
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=12,
        tiles="CartoDB positron",
        control_scale=True,
    )

    # Visual circle showing the search radius corridor
    folium.Circle(
        location=[center_lat, center_lon],
        radius=radius_km * 1000,
        color="#0284c7",
        weight=1.5,
        dash_array="5, 5",
        fill=True,
        fill_color="#38bdf8",
        fill_opacity=0.08,
        tooltip=f"Offset Corridor: {radius_km} km",
    ).add_to(m)

    # Layers for toggling wells on the map
    active_layer = folium.FeatureGroup(name="Current Well", show=True)
    suspected_layer = folium.FeatureGroup(
        name="nearby suspected wells", show=True
    )
    normal_layer = folium.FeatureGroup(name="normal wells", show=True)

    # 1. Target Current Well (Green marker)
    active_popup_html = f"""
    <div style="font-family: Arial; width: 220px; font-size:12px; color:#0f172a;">
        <div style="background:#16a34a; color:white; padding:6px 10px; border-radius:4px; font-weight:bold;">
            🎯 CURRENT WELL: {target_meta['well_id']}
        </div>
        <div style="padding-top:8px;">
            <b>Name:</b> {target_meta['well_name']}<br>
            <b>Field:</b> {target_meta['field']}<br>
            <b>Depth:</b> <span style="color:#16a34a; font-weight:bold;">{current_depth_val} m</span><br>
            <b>Formation:</b> <b>{curr_formation_val}</b><br>
            <b>Status:</b> {target_meta['well_status']}
        </div>
    </div>
    """
    folium.CircleMarker(
        location=[center_lat, center_lon],
        radius=13,
        color="#15803d",
        weight=3,
        fill=True,
        fill_color="#22c55e",
        fill_opacity=0.95,
        popup=folium.Popup(active_popup_html, max_width=300),
        tooltip=f"🎯 Current Well: {target_meta['well_id']} ({current_depth_val} m)",
    ).add_to(active_layer)

    # 2. Plot all nearby offset wells
    if not nearby_wells_df.empty:
        for _, well in nearby_wells_df.iterrows():
            w_id = str(well["well_id"])
            is_suspected = w_id in suspected_ids
            dist = well["distance_km"]

            w_norm = well["_id_norm"]
            w_events = events_df[events_df["_id_norm"] == w_norm]
            event_count = len(w_events)
            event_types_str = (
                ", ".join(w_events["event_type"].unique())
                if event_count > 0
                else "None"
            )

            # Red for suspected wells with incidents, Blue for normal offset wells
            if is_suspected:
                marker_color = "#dc2626"
                border_color = "#991b1b"
                radius = 10
                layer = suspected_layer
                status_text = (
                    f"<span style='color:#dc2626; font-weight:bold;'>⚠️ Suspected "
                    f"Hazard ({event_types_str})</span>"
                )
            else:
                marker_color = "#2563eb"
                border_color = "#1d4ed8"
                radius = 8
                layer = normal_layer
                status_text = (
                    "<span style='color:#2563eb; font-weight:bold;'>Normal "
                    "Offset</span>"
                )

            popup_html = f"""
            <div style="font-family: Arial; width: 230px; font-size:12px; color:#0f172a;">
                <div style="background:#1e293b; color:white; padding:6px 10px; border-radius:4px; font-weight:bold;">
                    {well['well_name']} ({w_id})
                </div>
                <div style="padding-top:8px;">
                    <b>Distance:</b> {dist} km<br>
                    <b>Formation:</b> {well['primary_formation']}<br>
                    <b>Status:</b> {well['well_status']}<br>
                    <b>Classification:</b> {status_text}<br>
                    <b>Total Incidents:</b> {event_count}
                </div>
            </div>
            """

            folium.CircleMarker(
                location=[well["latitude"], well["longitude"]],
                radius=radius,
                color=border_color,
                weight=2 if not is_suspected else 3,
                fill=True,
                fill_color=marker_color,
                fill_opacity=0.9,
                popup=folium.Popup(popup_html, max_width=320),
                tooltip=f"{w_id} ({dist} km) — {'🚨 Nearby Suspected Well' if is_suspected else 'Normal Well'}",
            ).add_to(layer)

    active_layer.add_to(m)
    suspected_layer.add_to(m)
    normal_layer.add_to(m)

    folium.LayerControl(collapsed=False, position="topright").add_to(m)
    return m


def create_offset_depth_profile(selected_meta, nearby_wells_df, nearby_events_df, bit_depth):
    """Build a subsurface cross-section with formation intervals and event depths."""
    current_well = pd.DataFrame([{
        "well_id": selected_meta["well_id"],
        "_id_norm": selected_meta["_id_norm"],
        "distance_km": 0.0,
    }])
    profile_wells = pd.concat(
        [current_well, nearby_wells_df[["well_id", "_id_norm", "distance_km"]]],
        ignore_index=True,
    ).drop_duplicates(subset=["_id_norm"])
    profile_wells = profile_wells.sort_values("distance_km")
    well_order = profile_wells["well_id"].tolist()

    formation_intervals = well_formations_df[
        well_formations_df["_id_norm"].isin(profile_wells["_id_norm"])
    ].merge(
        profile_wells[["_id_norm", "well_id", "distance_km"]],
        on="_id_norm",
        how="inner",
    )
    formation_intervals["top_depth_m"] = pd.to_numeric(
        formation_intervals["top_depth_m"], errors="coerce"
    )
    formation_intervals["bottom_depth_m"] = pd.to_numeric(
        formation_intervals["bottom_depth_m"], errors="coerce"
    )
    formation_intervals = formation_intervals.dropna(
        subset=["top_depth_m", "bottom_depth_m", "formation_name"]
    )

    event_points = nearby_events_df.copy()
    if not event_points.empty:
        event_points["depth_m"] = pd.to_numeric(event_points["depth_m"], errors="coerce")
        event_points = event_points.dropna(subset=["depth_m", "well_id", "event_type"])

    if formation_intervals.empty and event_points.empty:
        return None

    y_encoding = alt.Y(
        "top_depth_m:Q",
        title="Measured depth (m) — increasing downward",
        scale=alt.Scale(reverse=True, nice=True),
    )
    layers = []
    if not formation_intervals.empty:
        layers.append(
            alt.Chart(formation_intervals).mark_rect(opacity=0.48).encode(
                x=alt.X("well_id:N", title="Current well and nearby offset wells", sort=well_order),
                y=y_encoding,
                y2="bottom_depth_m:Q",
                color=alt.Color("formation_name:N", title="Formation"),
                tooltip=[
                    alt.Tooltip("well_id:N", title="Well"),
                    alt.Tooltip("distance_km:Q", title="Offset distance (km)", format=".2f"),
                    alt.Tooltip("formation_name:N", title="Formation"),
                    alt.Tooltip("top_depth_m:Q", title="Top depth (m)", format=".0f"),
                    alt.Tooltip("bottom_depth_m:Q", title="Bottom depth (m)", format=".0f"),
                ],
            )
        )
    if not event_points.empty:
        layers.append(
            alt.Chart(event_points).mark_point(filled=True, size=115, stroke="#0f172a", strokeWidth=0.7).encode(
                x=alt.X("well_id:N", title="Current well and nearby offset wells", sort=well_order),
                y=alt.Y("depth_m:Q", scale=alt.Scale(reverse=True), title="Measured depth (m) — increasing downward"),
                color=alt.Color(
                    "severity:N",
                    title="Event severity",
                    scale=alt.Scale(
                        domain=["Critical", "High", "Medium", "Low"],
                        range=["#991b1b", "#dc2626", "#f59e0b", "#2563eb"],
                    ),
                ),
                shape=alt.Shape("event_type:N", title="Event type"),
                tooltip=[
                    alt.Tooltip("well_id:N", title="Offset well"),
                    alt.Tooltip("distance_km:Q", title="Offset distance (km)", format=".2f"),
                    alt.Tooltip("depth_m:Q", title="Event depth (m)", format=".0f"),
                    alt.Tooltip("formation:N", title="Formation"),
                    alt.Tooltip("event_type:N", title="Event"),
                    alt.Tooltip("severity:N", title="Severity"),
                    alt.Tooltip("duration_hours:Q", title="Duration (hours)", format=".1f"),
                ],
            )
        )

    # This line places the operator's current bit depth against offset evidence.
    layers.append(
        alt.Chart(pd.DataFrame({"depth_m": [bit_depth]})).mark_rule(
            color="#16a34a", strokeDash=[7, 4], strokeWidth=2
        ).encode(y=alt.Y("depth_m:Q", scale=alt.Scale(reverse=True)))
    )
    return alt.layer(*layers).resolve_scale(color="independent").properties(height=520).interactive()


# Section 8: Main dashboard layout and risk analysis
st.markdown(
    """
    <div style="display: flex; justify-content: space-between; align-items: center; border-bottom: 2px solid #e2e8f0; padding-bottom: 8px; margin-bottom: 16px;">
        <div>
            <h1 style="margin: 0; font-size: 1.75rem; color: #0f172a; font-family: monospace;">🛢️ NWIS DASHBOARD</h1>
            <span style="font-size: 0.85rem; color: #64748b;">AI-Powered Offset Well Decision Support Platform</span>
        </div>
        <div>
            <span style="background: #e0f2fe; color: #0369a1; padding: 5px 14px; border-radius: 16px; font-size: 0.8rem; font-weight: 700; border: 1px solid #bae6fd;">
                ● eRTMAC Active
            </span>
        </div>
    </div>
""",
    unsafe_allow_html=True,
)

# 4 Key metrics across the top
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown(
        f"""
        <div class="nwis-card">
            <div class="metric-title">Current Well</div>
            <div class="metric-val">{selected_well_meta['well_id']} — {selected_well_meta['well_name'].split('-')[-1]}</div>
        </div>
    """,
        unsafe_allow_html=True,
    )

with c2:
    st.markdown(
        f"""
        <div class="nwis-card">
            <div class="metric-title">Depth</div>
            <div class="metric-val">{current_depth} m</div>
        </div>
    """,
        unsafe_allow_html=True,
    )

with c3:
    st.markdown(
        f"""
        <div class="nwis-card">
            <div class="metric-title">Formation</div>
            <div class="metric-val" style="color: #b45309;">{current_formation}</div>
        </div>
    """,
        unsafe_allow_html=True,
    )

with c4:
    st.markdown(
        f"""
        <div class="nwis-card">
            <div class="metric-title">Suspected Wells in Radius</div>
            <div class="metric-val" style="color: #dc2626;">{len(suspected_well_ids)} / {len(nearby_df)} Wells</div>
        </div>
    """,
        unsafe_allow_html=True,
    )

# Left column = Map, Right column = Risk Analysis & Historical Events
map_col, info_col = st.columns([1.1, 0.9])

with map_col:
    st.markdown("<div class='nwis-header'>MAP</div>", unsafe_allow_html=True)

    folium_map = create_nwis_map(
        selected_well_meta,
        nearby_df,
        suspected_well_ids,
        current_depth,
        current_formation,
    )
    map_html = folium_map._repr_html_()
    components.html(map_html, height=530, scrolling=False)

    # Map Legend
    st.markdown(
        f"""
        <div style="display: flex; gap: 24px; font-size: 0.82rem; color: #475569; margin-top: -10px; padding: 8px 14px; background: #ffffff; border: 1px solid #e2e8f0; border-radius: 6px;">
            <div><span style="color: #16a34a; font-size: 1.1rem;">●</span> <b>Current Well</b> ({selected_well_meta['well_id']})</div>
            <div><span style="color: #dc2626; font-size: 1.1rem;">●</span> <b>Nearby Suspected Wells</b> ({len(suspected_well_ids)})</div>
            <div><span style="color: #2563eb; font-size: 1.1rem;">●</span> <b>Normal Wells</b> ({len(nearby_df) - len(suspected_well_ids)})</div>
        </div>
    """,
        unsafe_allow_html=True,
    )

with info_col:
    st.markdown(
        "<div class='nwis-header'>DRILLING RISK ANALYSIS</div>",
        unsafe_allow_html=True,
    )

    hazard_label = ai_risk["top_hazard"]
    min_d, max_d = ai_risk["depth_range"]
    symp_count = ai_risk["matching_wells_count"]

    # Format the depth sentence cleanly
    if symp_count > 0:
        if min_d == max_d:
            depth_sentence = f"<b>{symp_count} nearby well(s)</b> had similar events at <b>{min_d} m</b>"
        else:
            depth_sentence = f"<b>{symp_count} nearby well(s)</b> had similar events around <b>{min_d}–{max_d} m</b>"
    else:
        depth_sentence = f"No direct incident recorded at this depth in offset wells; evaluated based on <i>{current_formation}</i> characteristics."

    # Risk Alert Banner
    st.markdown(
        f"""
        <div class="{ai_risk['badge_style']}">
            <div style="font-size: 1.2rem; font-weight: 800; font-family: monospace;">
                {ai_risk['level_display']}
            </div>
            <div style="font-size: 1.05rem; font-weight: 700; margin-top: 5px; color: #0f172a;">
                {hazard_label} risk detected
            </div>
            <div style="font-size: 0.92rem; margin-top: 8px; color: #334155; line-height: 1.4;">
                {depth_sentence}
            </div>
        </div>
    """,
        unsafe_allow_html=True,
    )

    # Grouped Historical Events by Well ID with DDR Report Links
    st.markdown(
        "<div class='nwis-header'>HISTORICAL EVENTS (OFFSET WELLS)</div>",
        unsafe_allow_html=True,
    )

    if not offset_events.empty:
        grouped_wells = offset_events.groupby("well_id")

        st.markdown(
            '<div class="scrollable-events-container">', unsafe_allow_html=True
        )

        for w_id, w_ev_df in grouped_wells:
            is_sus = w_id in suspected_well_ids
            dist_val = w_ev_df.iloc[0]["distance_km"]
            report_name = f"{w_id}_DDR.pdf"

            card_border = "#dc2626" if is_sus else "#cbd5e1"
            tag_label = "SUSPECTED HAZARD" if is_sus else "OFFSET LOG"
            tag_bg = "#fef2f2" if is_sus else "#f1f5f9"
            tag_color = "#dc2626" if is_sus else "#475569"

            with st.container():
                st.markdown(
                    f"""
                    <div style="border: 1px solid {card_border}; background: #ffffff; border-radius: 6px; padding: 10px 14px; margin-bottom: 10px;">
                        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
                            <span style="font-weight: 700; font-size: 0.95rem; color: #0f172a;">Well {w_id} ({dist_val} km)</span>
                            <span style="background: {tag_bg}; color: {tag_color}; font-size: 0.72rem; font-weight: 700; padding: 2px 8px; border-radius: 4px; border: 1px solid {card_border};">
                                {tag_label}
                            </span>
                        </div>
                    """,
                    unsafe_allow_html=True,
                )

                # Show all incidents recorded for this specific offset well
                for _, ev in w_ev_df.iterrows():
                    ev_type = ev["event_type"]
                    ev_depth = ev["depth_m"]
                    ev_sev = ev["severity"]
                    st.markdown(
                        f"""
                        <div style="font-size: 0.85rem; color: #334155; padding-left: 8px; margin-bottom: 4px;">
                            • <b>{ev_type}</b> at <b>{ev_depth} m</b> ({ev_sev} Severity) — <i>{ev['operational_impact']}</i>
                        </div>
                    """,
                        unsafe_allow_html=True,
                    )

                # The old #filename anchor was not a real file URL. Streamlit's
                # download button reliably serves the actual project PDF instead.
                report_path = resolve_report_path(w_id, "DDR")
                st.markdown(
                    '<div style="margin-top: 8px; padding-top: 6px; border-top: 1px dashed #e2e8f0; font-size: 0.82rem;">'
                    '<b>📄 Relevant Report</b></div>',
                    unsafe_allow_html=True,
                )
                if report_path:
                    st.download_button(
                        label=f"Open / download {report_name}",
                        data=load_report_file(report_path),
                        file_name=os.path.basename(report_path),
                        mime="application/pdf",
                        key=f"report-download-{w_id}",
                        use_container_width=True,
                    )
                else:
                    st.caption(f"No local DDR PDF was found for {w_id}.")
                st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.info("No historical drilling events found in the offset radius.")


# Section 9: Offset-well subsurface structure and depth-event profile
st.markdown(
    "<div class='nwis-header'>OFFSET-WELL SUBSURFACE EVENT PROFILE</div>",
    unsafe_allow_html=True,
)
st.caption(
    "Formation bands show the subsurface intervals for the current well and nearby offsets. "
    "Event markers show the historical event type and severity at its recorded depth. "
    "The green dashed line is the current bit depth. This is historical offset evidence, not a prediction."
)
depth_profile = create_offset_depth_profile(
    selected_well_meta, nearby_df, offset_events, current_depth
)
if depth_profile is None:
    st.info("No formation intervals or historical event depths are available for the selected offset corridor.")
else:
    st.altair_chart(depth_profile, use_container_width=True)


# Section 10: Detailed tables and model inspection
with st.expander("📊 Offset Well Data Table & Model Details"):
    tab1, tab2, tab3 = st.tabs(
        ["Corridor Offset Wells", "Incident Catalog", "ML Feature Importance"]
    )

    with tab1:
        if not nearby_df.empty:
            st.dataframe(
                nearby_df[
                    [
                        "well_id",
                        "well_name",
                        "field",
                        "distance_km",
                        "total_depth_m",
                        "primary_formation",
                        "well_status",
                    ]
                ],
                use_container_width=True,
            )
        else:
            st.write("No nearby offset wells found.")

    with tab2:
        if not offset_events.empty:
            st.dataframe(
                offset_events[
                    [
                        "event_id",
                        "well_id",
                        "distance_km",
                        "depth_m",
                        "formation",
                        "event_type",
                        "severity",
                        "duration_hours",
                    ]
                ],
                use_container_width=True,
            )
        else:
            st.write("No events recorded for this subset.")

    with tab3:
        # Displays the relative weight the Random Forest model places on each parameter
        fi_df = pd.DataFrame(
            {
                "Drilling Parameter": feature_cols,
                "Importance Weight (%)": np.round(
                    ml_model.feature_importances_ * 100, 2
                ),
            }
        ).sort_values(by="Importance Weight (%)", ascending=False)
        st.dataframe(fi_df, use_container_width=True)


# Section 10: Data-grounded well information chatbot
st.divider()
st.markdown("<div class='nwis-header'>WELL INFORMATION ASSISTANT</div>", unsafe_allow_html=True)
st.caption(
    "Ask about the selected well, offset wells, formations, historical events, "
    "or the current risk evidence. Responses are grounded only in loaded dashboard data."
)

if "well_chat_messages" not in st.session_state:
    st.session_state.well_chat_messages = [
        {
            "role": "assistant",
            "content": "Ask me about the selected well, nearby offsets, historical events, or current risk evidence.",
        }
    ]

chat_col, clear_col = st.columns([0.86, 0.14])
with clear_col:
    if st.button("Clear chat", use_container_width=True):
        st.session_state.well_chat_messages = [
            {
                "role": "assistant",
                "content": "Chat cleared. Ask about the currently selected well.",
            }
        ]
        st.rerun()

with chat_col:
    if not os.getenv("OPENAI_API_KEY"):
        st.warning("Chatbot disabled: add OPENAI_API_KEY to .env, then restart Streamlit.")
    else:
        for message in st.session_state.well_chat_messages:
            with st.chat_message(message["role"]):
                st.write(message["content"])

        question = st.chat_input("Example: What historical risks are near the current depth?")
        if question:
            st.session_state.well_chat_messages.append({"role": "user", "content": question})
            with st.chat_message("user"):
                st.write(question)

            context = build_well_chat_context(
                selected_well_meta,
                nearby_df,
                offset_events,
                current_formation,
                current_depth,
                {
                    "mud_weight_ppg": live_mw,
                    "rop_m_per_hr": live_rop,
                    "rpm": live_rpm,
                    "wob_klbf": live_wob,
                },
                ai_risk,
            )
            with st.chat_message("assistant"):
                with st.spinner("Reviewing selected-well evidence..."):
                    try:
                        answer = ask_well_chatbot(
                            question, context, st.session_state.well_chat_messages[:-1]
                        )
                    except Exception as exc:
                        answer = f"I could not answer that request: {exc}"
                st.write(answer)
            st.session_state.well_chat_messages.append({"role": "assistant", "content": answer})
