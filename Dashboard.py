"""Flask dashboard for nearby-well intelligence and offset-risk evidence."""
from __future__ import annotations

import json
import os
import re
import unicodedata
from pathlib import Path
from typing import Any

import altair as alt
import folium
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from flask import Flask, abort, jsonify, render_template, request, send_file, session
from geopy.distance import geodesic
from openai import OpenAI
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

ROOT_DIR = Path(__file__).resolve().parent
DATASET_DIR = ROOT_DIR / "datasets"
REPORT_DIR = ROOT_DIR / "data"
load_dotenv(ROOT_DIR / ".env")

app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "change-this-flask-secret-before-deployment")
CHAT_MODEL = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")


def normalize_well_id(value: Any) -> str:
    """Create a matching-only well identifier without changing source IDs."""
    if pd.isna(value):
        return ""
    value = unicodedata.normalize("NFKC", str(value))
    value = "".join(char for char in value if not char.isspace() and unicodedata.category(char) != "Cf")
    return re.sub(r"[-_\u2010-\u2015]", "", value.strip().upper())


def load_datasets() -> tuple[pd.DataFrame, ...]:
    """Load the five existing synthetic datasets and add matching helper IDs."""
    wells = pd.read_csv(DATASET_DIR / "wells.csv")
    events = pd.read_csv(DATASET_DIR / "drilling_events.csv")
    parameters = pd.read_csv(DATASET_DIR / "drilling_parameters.csv")
    formations = pd.read_csv(DATASET_DIR / "formations.csv")
    well_formations = pd.read_csv(DATASET_DIR / "well_formations.csv")
    wells["latitude"] = pd.to_numeric(wells["latitude"], errors="coerce")
    wells["longitude"] = pd.to_numeric(wells["longitude"], errors="coerce")
    wells = wells.dropna(subset=["latitude", "longitude"]).copy()
    for frame in (wells, events, parameters, well_formations):
        frame["_id_norm"] = frame["well_id"].apply(normalize_well_id)
    return wells, events, parameters, formations, well_formations


WELLS, EVENTS, PARAMETERS, FORMATIONS, WELL_FORMATIONS = load_datasets()


def formation_at_depth(well_norm: str, depth: float) -> str:
    rows = WELL_FORMATIONS[WELL_FORMATIONS["_id_norm"] == well_norm]
    if rows.empty:
        return "Unknown Formation"
    matched = rows[(rows["top_depth_m"] <= depth) & (rows["bottom_depth_m"] >= depth)]
    if not matched.empty:
        return str(matched.iloc[0]["formation_name"])
    return str(rows.sort_values("top_depth_m" if depth < rows["top_depth_m"].min() else "bottom_depth_m").iloc[0]["formation_name"])


def nearby_wells(well_norm: str, radius_km: float) -> pd.DataFrame:
    selected = WELLS[WELLS["_id_norm"] == well_norm]
    if selected.empty:
        return pd.DataFrame()
    location = (selected.iloc[0]["latitude"], selected.iloc[0]["longitude"])
    output: list[dict[str, Any]] = []
    for _, well in WELLS.iterrows():
        if well["_id_norm"] == well_norm:
            continue
        distance = geodesic(location, (well["latitude"], well["longitude"])).km
        if distance <= radius_km:
            output.append({
                "well_id": well["well_id"], "_id_norm": well["_id_norm"], "well_name": well["well_name"],
                "field": well["field"], "latitude": well["latitude"], "longitude": well["longitude"],
                "total_depth_m": well["total_depth_m"], "primary_formation": well["primary_formation"],
                "well_status": well["well_status"], "distance_km": round(distance, 2),
            })
    return pd.DataFrame(output).sort_values("distance_km").reset_index(drop=True) if output else pd.DataFrame()


def train_model() -> tuple[Any, LabelEncoder, LabelEncoder, list[str]]:
    records = PARAMETERS.copy()
    # Label records in batches. The former Streamlit implementation applied a
    # DataFrame filter for every parameter row, making Flask startup very slow.
    records["formation"] = "Unknown Formation"
    for _, interval in WELL_FORMATIONS.iterrows():
        match = ((records["_id_norm"] == interval["_id_norm"])
                 & (records["depth_m"] >= interval["top_depth_m"])
                 & (records["depth_m"] <= interval["bottom_depth_m"]))
        records.loc[match, "formation"] = interval["formation_name"]
    records["event_label"] = "Normal"
    for _, event in EVENTS.iterrows():
        match = ((records["_id_norm"] == event["_id_norm"])
                 & (abs(records["depth_m"] - event["depth_m"]) <= 50)
                 & (records["event_label"] == "Normal"))
        records.loc[match, "event_label"] = event["event_type"]
    form_encoder, event_encoder = LabelEncoder(), LabelEncoder()
    records["formation_encoded"] = form_encoder.fit_transform(records["formation"])
    records["event_encoded"] = event_encoder.fit_transform(records["event_label"])
    features = ["depth_m", "mud_weight_ppg", "rop_m_per_hr", "rpm", "wob_klbf", "formation_encoded"]
    model = RandomForestClassifier(n_estimators=80, max_depth=12, random_state=42, class_weight="balanced", n_jobs=-1)
    model.fit(records[features].fillna(0), records["event_encoded"])
    return model, form_encoder, event_encoder, features


MODEL, FORM_ENCODER, EVENT_ENCODER, FEATURES = train_model()


def risk_assessment(depth: float, telemetry: dict[str, float], formation: str, offset_events: pd.DataFrame) -> dict[str, Any]:
    formation_code = FORM_ENCODER.transform([formation])[0] if formation in FORM_ENCODER.classes_ else 0
    values = {"depth_m": depth, "formation_encoded": formation_code, **telemetry}
    probabilities = MODEL.predict_proba(pd.DataFrame([values])[FEATURES])[0]
    risk_map = {EVENT_ENCODER.classes_[index]: probability for index, probability in enumerate(probabilities)}
    hazards = {name: score for name, score in risk_map.items() if name != "Normal"}
    hazard = max(hazards, key=hazards.get) if hazards else "Mud Loss"
    score = hazards.get(hazard, 0.0)
    similar = offset_events[(abs(offset_events["depth_m"] - depth) <= 150) | (offset_events["formation"].str.casefold() == formation.casefold())] if not offset_events.empty else pd.DataFrame()
    critical = not similar.empty and (similar["severity"].str.casefold() == "critical").any()
    high = not similar.empty and (similar["severity"].str.casefold() == "high").any()
    count = int(similar["well_id"].nunique()) if not similar.empty else 0
    if not similar.empty and not similar["event_type"].mode().empty:
        hazard = str(similar["event_type"].mode().iloc[0])
    if critical or score > .65:
        level, css = "CATASTROPHIC", "critical"
    elif high or count >= 2 or score > .35:
        level, css = "HIGH", "high"
    elif count == 1 or score > .20:
        level, css = "CAUTIONARY", "caution"
    else:
        level, css = "LOW", "low"
    return {"level": level, "css": css, "hazard": hazard, "confidence": round(score * 100, 1), "matching_wells_count": count,
            "depth_range": (int(similar["depth_m"].min()), int(similar["depth_m"].max())) if not similar.empty else (int(depth), int(depth)), "similar": similar}


def make_map(selected: pd.Series, nearby: pd.DataFrame, suspected: set[str], radius: float) -> str:
    # OpenStreetMap is a public Folium basemap and does not require a CARTO/API key.
    # It also keeps the dashboard usable when CARTO tile access is restricted.
    fmap = folium.Map(location=[selected["latitude"], selected["longitude"]], zoom_start=12, tiles="OpenStreetMap", control_scale=True)
    folium.Circle(location=[selected["latitude"], selected["longitude"]], radius=radius * 1000, color="#0284c7", weight=1.5, dash_array="5,5", fill=True, fill_opacity=.08).add_to(fmap)
    folium.CircleMarker(location=[selected["latitude"], selected["longitude"]], radius=13, color="#15803d", fill=True, fill_color="#22c55e", fill_opacity=.95, tooltip=f"Current well: {selected['well_id']}").add_to(fmap)
    for _, well in nearby.iterrows():
        suspected_well = well["well_id"] in suspected
        color = "#dc2626" if suspected_well else "#2563eb"
        events = EVENTS[EVENTS["_id_norm"] == well["_id_norm"]]
        popup = f"<b>{well['well_name']} ({well['well_id']})</b><br>Distance: {well['distance_km']} km<br>Formation: {well['primary_formation']}<br>Incidents: {len(events)}"
        folium.CircleMarker(location=[well["latitude"], well["longitude"]], radius=10 if suspected_well else 8, color=color, fill=True, fill_color=color, fill_opacity=.9, popup=popup, tooltip=f"{well['well_id']} ({well['distance_km']} km)").add_to(fmap)
    return fmap._repr_html_()


def depth_profile(selected: pd.Series, nearby: pd.DataFrame, events: pd.DataFrame, depth: float) -> dict[str, Any] | None:
    profile = pd.concat([pd.DataFrame([{"well_id": selected["well_id"], "_id_norm": selected["_id_norm"], "distance_km": 0.0}]), nearby[["well_id", "_id_norm", "distance_km"]]], ignore_index=True)
    profile = profile.drop_duplicates("_id_norm").sort_values("distance_km")
    intervals = WELL_FORMATIONS[WELL_FORMATIONS["_id_norm"].isin(profile["_id_norm"])].merge(
        profile, on="_id_norm", suffixes=("", "_profile")
    )
    if intervals.empty and events.empty:
        return None
    order = profile["well_id"].tolist()
    charts = []
    if not intervals.empty:
        charts.append(alt.Chart(intervals).mark_rect(opacity=.48).encode(x=alt.X("well_id:N", sort=order, title="Current and offset wells"), y=alt.Y("top_depth_m:Q", scale=alt.Scale(reverse=True), title="Measured depth (m)"), y2="bottom_depth_m:Q", color=alt.Color("formation_name:N", title="Formation"), tooltip=["well_id", "distance_km", "formation_name", "top_depth_m", "bottom_depth_m"]))
    if not events.empty:
        charts.append(alt.Chart(events).mark_point(filled=True, size=110, stroke="#0f172a").encode(x=alt.X("well_id:N", sort=order, title="Current and offset wells"), y=alt.Y("depth_m:Q", scale=alt.Scale(reverse=True), title="Measured depth (m)"), color=alt.Color("severity:N", scale=alt.Scale(domain=["Critical", "High", "Medium", "Low"], range=["#991b1b", "#dc2626", "#f59e0b", "#2563eb"])), shape="event_type:N", tooltip=["well_id", "distance_km", "depth_m", "formation", "event_type", "severity", "duration_hours"]))
    charts.append(alt.Chart(pd.DataFrame({"depth_m": [depth]})).mark_rule(color="#16a34a", strokeDash=[7, 4], strokeWidth=2).encode(y=alt.Y("depth_m:Q", scale=alt.Scale(reverse=True))))
    return json.loads(alt.layer(*charts).resolve_scale(color="independent").properties(height=520).interactive().to_json())


def report_path(well_id: str, report_type: str) -> Path | None:
    if report_type not in {"DDR", "WCR"} or not re.fullmatch(r"[A-Za-z0-9_-]+", well_id):
        return None
    candidates = [REPORT_DIR / "reports" / f"{well_id}_{report_type}.pdf", REPORT_DIR / "scanned_reports" / f"{well_id}_{report_type}.pdf", REPORT_DIR / "scanned_reports" / f"{well_id}_{report_type}_scanned.pdf"]
    return next((path for path in candidates if path.is_file()), None)


def json_safe(value: Any) -> Any:
    """Convert scalar Pandas values while preserving lists/tuples for JSON context."""
    if isinstance(value, np.generic):
        return value.item()
    if value is None:
        return None
    if isinstance(value, float) and pd.isna(value):
        return None
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


def chat_context(selected: pd.Series, nearby: pd.DataFrame, events: pd.DataFrame, formation: str, depth: float, telemetry: dict[str, float], risk: dict[str, Any]) -> dict[str, Any]:
    fields = ["well_id", "well_name", "latitude", "longitude", "field", "total_depth_m", "primary_formation", "well_status"]
    return {"current_well": {key: json_safe(selected[key]) for key in fields}, "operational_context": {"bit_depth_m": depth, "formation_at_depth": formation, "telemetry": telemetry, "risk": {key: json_safe(value) for key, value in risk.items() if key != "similar"}}, "nearby_wells": nearby.reindex(columns=["well_id", "well_name", "distance_km", "field", "primary_formation"]).head(30).to_dict("records"), "historical_offset_events": events.reindex(columns=["well_id", "distance_km", "depth_m", "formation", "event_type", "severity", "duration_hours", "operational_impact"]).head(80).to_dict("records")}


@app.get("/")
def dashboard():
    well_id = request.args.get("well", "W141")
    selected = WELLS[WELLS["_id_norm"] == normalize_well_id(well_id)]
    selected = selected.iloc[0] if not selected.empty else WELLS.iloc[0]
    radius = float(request.args.get("radius", 10))
    radius = min(max(radius, 3), 30)
    max_depth = int(selected["total_depth_m"])
    depth = min(max(float(request.args.get("depth", min(2790, max_depth))), 25), max_depth)
    matching = PARAMETERS[(PARAMETERS["_id_norm"] == selected["_id_norm"]) & (PARAMETERS["depth_m"] <= depth)]
    defaults = matching.iloc[-1] if not matching.empty else pd.Series({"mud_weight_ppg": 12.6, "rop_m_per_hr": 17.4, "rpm": 105.0, "wob_klbf": 25.7})
    telemetry = {key: float(request.args.get(key, defaults[key])) for key in ("mud_weight_ppg", "rop_m_per_hr", "rpm", "wob_klbf")}
    formation = formation_at_depth(selected["_id_norm"], depth)
    nearby = nearby_wells(selected["_id_norm"], radius)
    events = EVENTS[EVENTS["_id_norm"].isin(nearby["_id_norm"])].copy() if not nearby.empty else pd.DataFrame(columns=EVENTS.columns)
    if not events.empty:
        events = events.merge(nearby[["_id_norm", "distance_km", "well_name"]], on="_id_norm", how="left")
    risk = risk_assessment(depth, telemetry, formation, events)
    suspected = set(risk["similar"]["well_id"].tolist()) if not risk["similar"].empty else set()
    context = chat_context(selected, nearby, events, formation, depth, telemetry, risk)
    session["chat_context"] = context
    return render_template("dashboard.html", wells=WELLS["well_id"].tolist(), selected=selected, radius=radius, depth=depth, max_depth=max_depth, telemetry=telemetry, formation=formation, nearby=nearby.to_dict("records"), events=events.to_dict("records"), risk=risk, suspected=suspected, map_html=make_map(selected, nearby, suspected, radius), profile_spec=depth_profile(selected, nearby, events, depth), feature_importance=pd.DataFrame({"feature": FEATURES, "importance": np.round(MODEL.feature_importances_ * 100, 2)}).sort_values("importance", ascending=False).to_dict("records"), chat_enabled=bool(os.getenv("OPENAI_API_KEY")))


@app.get("/report/<well_id>/<report_type>")
def download_report(well_id: str, report_type: str):
    path = report_path(well_id, report_type.upper())
    if path is None:
        abort(404, "Report not found")
    return send_file(path, mimetype="application/pdf", as_attachment=False, download_name=path.name)


@app.post("/api/chat")
def chat():
    if not os.getenv("OPENAI_API_KEY"):
        return jsonify(error="OPENAI_API_KEY is missing."), 503
    question = (request.get_json(silent=True) or {}).get("question", "").strip()
    if not question:
        return jsonify(error="A question is required."), 400
    history = session.get("chat_history", [])[-6:]
    prompt = "You are the NWIS Well Information Assistant. Use ONLY the supplied evidence. Never invent wells, depths, formations, incidents, or risks. If information is missing, say so. Be concise and state this does not replace approved drilling procedures."
    try:
        answer = OpenAI().responses.create(model=CHAT_MODEL, instructions=prompt, input=history + [{"role": "user", "content": "EVIDENCE:\n" + json.dumps(session.get("chat_context", {}), default=str) + "\n\nQUESTION:\n" + question}], store=False).output_text.strip()
    except Exception as exc:
        return jsonify(error=f"Chatbot request failed: {exc}"), 502
    session["chat_history"] = (history + [{"role": "user", "content": question}, {"role": "assistant", "content": answer}])[-8:]
    return jsonify(answer=answer)


@app.post("/api/chat/clear")
def clear_chat():
    session.pop("chat_history", None)
    return jsonify(ok=True)


if __name__ == "__main__":
    # Keep the dev reloader off: it would train the existing risk model twice.
    app.run(host="127.0.0.1", port=5000, debug=False)
