# PS 26121 — Oil & Gas Offset-Well Risk Monitoring

AI-powered system for identifying potential drilling risks using historical offset-well data.

## 📊 Current Progress

The project uses **5 connected CSV datasets**:

- `wells.csv` — Well location and basic information
- `drilling_events.csv` — Historical drilling problems
- `drilling_parameters.csv` — Drilling conditions
- `formations.csv` — Formation information
- `well_formations.csv` — Formations encountered by each well at different depths

## ✅ Current Implementation

- Load `wells.csv` using Pandas.
- Plot well coordinates using Folium.
- Add interactive well markers, popups, search, clustering, and status-based colors.
- Select a well and find nearby offset wells using geographic distance.
- Connect nearby wells with `drilling_events.csv` using `well_id`.
- Retrieve historical events from nearby wells.

## 🔄 Planned Workflow

```text
wells.csv
   ↓
Folium Well Map
   ↓
Select Current Well
   ↓
Find Nearby Offset Wells
   ↓
well_formations.csv
   ↓
Match Formation + Depth
   ↓
drilling_events.csv
   ↓
Historical Risk Evidence
   ↓
drilling_parameters.csv
   ↓
Data Analysis / ML
   ↓
Risk Prediction + Alert