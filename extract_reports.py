import os
import json
import pdfplumber
import pandas as pd
from openai import OpenAI

# =====================================================================
# 🔑 PASTE YOUR OPENAI API KEY HERE BEFORE RUNNING
# (Make sure to remove your real key before committing/pushing to GitHub!)
# =====================================================================
OPENAI_API_KEY = "PASTE_YOUR_OPENAI_API_KEY_HERE"

# Validate that the API key was inserted
if not OPENAI_API_KEY or OPENAI_API_KEY == "PASTE_YOUR_OPENAI_API_KEY_HERE":
    raise ValueError(
        "❌ Error: Please paste your OpenAI API key into the OPENAI_API_KEY variable at the top of extract_reports.py"
    )

# Initialize OpenAI Client directly with the provided key
client = OpenAI(api_key=OPENAI_API_KEY)


def extract_raw_text_from_pdf(pdf_path: str) -> str:
    """Extracts all text from pages and tables in the PDF."""
    full_text = []
    with pdfplumber.open(pdf_path) as pdf:
        for page_idx, page in enumerate(pdf.pages, start=1):
            text = page.extract_text()
            if text:
                full_text.append(f"--- PAGE {page_idx} ---\n" + text)
    return "\n\n".join(full_text)


def parse_report_with_openai(raw_text: str, filename: str) -> dict:
    """Passes raw text to OpenAI to structure into the required 5 CSV schemas."""
    
    system_prompt = """
    You are an expert petroleum data extraction assistant for Oil India completion and daily drilling reports.
    Extract data and return ONLY a valid JSON object matching the requested schema.
    
    Expected JSON Structure:
    {
      "wells": [
        {
          "well_id": "string",
          "well_name": "string",
          "latitude": float,
          "longitude": float,
          "total_depth_m": float,
          "field": "string",
          "primary_formation": "string",
          "well_status": "string"
        }
      ],
      "drilling_events": [
        {
          "event_id": "string",
          "well_id": "string",
          "depth_m": float,
          "formation": "string",
          "event_type": "string",
          "severity": "string",
          "event_description": "string",
          "duration_hours": float,
          "operational_impact": "string"
        }
      ],
      "drilling_parameters": [
        {
          "well_id": "string",
          "depth_m": float,
          "mud_weight_ppg": float,
          "rop_m_per_hr": float,
          "rpm": float,
          "wob_klbf": float
        }
      ],
      "formations": [
        {
          "formation_id": "string",
          "formation_name": "string",
          "formation_type": "string",
          "typical_risk": "string",
          "description": "string"
        }
      ],
      "well_formations": [
        {
          "well_id": "string",
          "formation_name": "string",
          "top_depth_m": float,
          "bottom_depth_m": float
        }
      ]
    }
    
    Instructions:
    - If a specific table or section is not in the current document, return an empty list `[]` for that key.
    - Clean numbers (e.g., '3,450 m' -> 3450.0).
    - Strip units from numeric values.
    - Maintain consistent well_id across all tables (e.g., 'W001', 'W002').
    """

    user_prompt = f"Extract all relevant entities from this document ({filename}):\n\n{raw_text}"

    response = client.chat.completions.create(
        model="gpt-4o",  # or "gpt-4o-mini"
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.0
    )

    content = response.choices[0].message.content
    return json.loads(content)


def process_all_reports(input_folder: str, output_folder: str):
    """Main processing loop."""
    os.makedirs(output_folder, exist_ok=True)
    
    aggregated_data = {
        "wells": [],
        "drilling_events": [],
        "drilling_parameters": [],
        "formations": [],
        "well_formations": []
    }

    pdf_files = [f for f in os.listdir(input_folder) if f.lower().endswith(".pdf")]
    
    if not pdf_files:
        print(f"No PDF files found in {input_folder}")
        return

    for filename in sorted(pdf_files):
        pdf_path = os.path.join(input_folder, filename)
        print(f"🔄 Processing: {filename}...")
        
        try:
            raw_text = extract_raw_text_from_pdf(pdf_path)
            extracted_json = parse_report_with_openai(raw_text, filename)
            
            for key in aggregated_data.keys():
                items = extracted_json.get(key, [])
                if items:
                    aggregated_data[key].extend(items)
                    
            print(f"✅ Successfully processed {filename}")
        except Exception as e:
            print(f"❌ Error processing {filename}: {e}")

    # Build and clean DataFrames
    print("\n📦 Generating consolidated CSV files...")

    # 1. Wells CSV
    df_wells = pd.DataFrame(aggregated_data["wells"])
    if not df_wells.empty:
        df_wells = df_wells.drop_duplicates(subset=["well_id"], keep="last")
    df_wells.to_csv(os.path.join(output_folder, "wells.csv"), index=False)

    # 2. Drilling Events CSV
    df_events = pd.DataFrame(aggregated_data["drilling_events"])
    if not df_events.empty:
        df_events = df_events.drop_duplicates(subset=["event_id", "well_id"], keep="first")
    df_events.to_csv(os.path.join(output_folder, "drilling_events.csv"), index=False)

    # 3. Drilling Parameters CSV
    df_params = pd.DataFrame(aggregated_data["drilling_parameters"])
    if not df_params.empty:
        df_params = df_params.drop_duplicates(subset=["well_id", "depth_m"], keep="last")
    df_params.to_csv(os.path.join(output_folder, "drilling_parameters.csv"), index=False)

    # 4. Formations CSV
    df_formations = pd.DataFrame(aggregated_data["formations"])
    if not df_formations.empty:
        df_formations = df_formations.drop_duplicates(subset=["formation_name"], keep="first")
    df_formations.to_csv(os.path.join(output_folder, "formations.csv"), index=False)

    # 5. Well Formations CSV
    df_well_formations = pd.DataFrame(aggregated_data["well_formations"])
    if not df_well_formations.empty:
        df_well_formations = df_well_formations.drop_duplicates(subset=["well_id", "formation_name", "top_depth_m"], keep="first")
    df_well_formations.to_csv(os.path.join(output_folder, "well_formations.csv"), index=False)

    print(f"🎉 Complete! All CSV files saved to folder: '{output_folder}'")


if __name__ == "__main__":
    INPUT_DIR = "./input_pdfs"
    OUTPUT_DIR = "./output_csvs"
    process_all_reports(INPUT_DIR, OUTPUT_DIR)