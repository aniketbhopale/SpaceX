"""Extract DDR/WCR PDF reports into CSVs compatible with the synthetic datasets.

The program works without an API key in local mode.  Install Tesseract separately
for scanned PDFs; on Windows set TESSERACT_CMD below when it is not on PATH.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import re
import shutil
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any, Optional

import pandas as pd
from pydantic import BaseModel, Field, ValidationError

try:
    from dotenv import load_dotenv
except ImportError:  # Helpful when a user has not installed optional OpenAI support.
    load_dotenv = None


# Configuration. Tesseract is an operating-system installation, not a pip package.
MIN_TEXT_CHARS = 50
OCR_DPI = 250
OCR_THRESHOLD = False
TESSERACT_CMD: Optional[str] = None
DEFAULT_INPUT_DIR = "./data"
DEFAULT_DATASET_DIR = "./dataset" if Path("./dataset").exists() else "./datasets"
DEFAULT_OUTPUT_DIR = "./output"
TABLE_MARKER = "--- TABLE EXTRACTION ---"

DATASET_KEYS = {
    "wells": "wells.csv",
    "well_formations": "well_formations.csv",
    "formations": "formations.csv",
    "drilling_parameters": "drilling_parameters.csv",
    "drilling_events": "drilling_events.csv",
}
LOGGER = logging.getLogger("petroleum_report_extractor")


# These Pydantic models reflect the actual current synthetic CSV columns. Every
# field is optional because a report must not invent data it does not contain.
class WellRecord(BaseModel):
    well_id: Optional[str] = None
    well_name: Optional[str] = None
    latitude: Optional[float] = Field(None, ge=-90, le=90)
    longitude: Optional[float] = Field(None, ge=-180, le=180)
    field: Optional[str] = None
    total_depth_m: Optional[float] = Field(None, ge=0)
    primary_formation: Optional[str] = None
    well_status: Optional[str] = None
    spud_date: Optional[str] = None
    well_role: Optional[str] = None


class WellFormationRecord(BaseModel):
    well_id: Optional[str] = None
    formation_name: Optional[str] = None
    top_depth_m: Optional[float] = Field(None, ge=0)
    bottom_depth_m: Optional[float] = Field(None, ge=0)


class FormationRecord(BaseModel):
    formation_id: Optional[str] = None
    formation_name: Optional[str] = None
    formation_type: Optional[str] = None
    typical_risk: Optional[str] = None
    description: Optional[str] = None


class DrillingParameterRecord(BaseModel):
    record_id: Optional[str] = None
    well_id: Optional[str] = None
    depth_m: Optional[float] = Field(None, ge=0)
    mud_weight_ppg: Optional[float] = Field(None, ge=0)
    rop_m_per_hr: Optional[float] = Field(None, ge=0)
    rpm: Optional[float] = Field(None, ge=0)
    wob_klbf: Optional[float] = Field(None, ge=0)


class DrillingEventRecord(BaseModel):
    event_id: Optional[str] = None
    well_id: Optional[str] = None
    depth_m: Optional[float] = Field(None, ge=0)
    formation: Optional[str] = None
    event_type: Optional[str] = None
    severity: Optional[str] = None
    event_description: Optional[str] = None
    duration_hours: Optional[float] = Field(None, ge=0)
    operational_impact: Optional[str] = None


class ExtractionPayload(BaseModel):
    wells: list[WellRecord] = Field(default_factory=list)
    well_formations: list[WellFormationRecord] = Field(default_factory=list)
    formations: list[FormationRecord] = Field(default_factory=list)
    drilling_parameters: list[DrillingParameterRecord] = Field(default_factory=list)
    drilling_events: list[DrillingEventRecord] = Field(default_factory=list)


MODEL_BY_KEY = {
    "wells": WellRecord, "well_formations": WellFormationRecord,
    "formations": FormationRecord, "drilling_parameters": DrillingParameterRecord,
    "drilling_events": DrillingEventRecord,
}


def load_dataset_schema(dataset_dir: Path) -> tuple[dict[str, list[str]], dict[str, pd.DataFrame]]:
    """Load source schemas and reference values without modifying the datasets."""
    schemas, reference = {}, {}
    for key, filename in DATASET_KEYS.items():
        path = dataset_dir / filename
        if not path.exists():
            raise FileNotFoundError(f"Required synthetic dataset is missing: {path}")
        frame = pd.read_csv(path, dtype="string")
        schemas[key] = frame.columns.tolist()
        reference[key] = frame
    return schemas, reference


def detect_pdf_type(pdf_path: Path, min_text_chars: int = MIN_TEXT_CHARS) -> str:
    """Classify by usable embedded text, not by whether pages contain images."""
    try:
        import pymupdf as fitz
    except ImportError as exc:
        raise RuntimeError("PyMuPDF is required for PDF type detection. Run: pip install pymupdf") from exc
    text_parts: list[str] = []
    with fitz.open(pdf_path) as document:
        for page in document:
            try:
                text_parts.append(page.get_text("text") or "")
            except Exception as exc:  # A bad page should not prevent classification.
                LOGGER.warning("Text inspection failed on %s: %s", pdf_path.name, exc)
    usable = len(re.sub(r"\s+", "", "".join(text_parts)))
    pdf_type = "DIGITAL" if usable >= min_text_chars else "SCANNED"
    print(f"[PDF TYPE] {pdf_path.name} -> {pdf_type}")
    return pdf_type


def extract_text_from_digital_pdf(pdf_path: Path) -> str:
    """Extract every page with clear page boundaries using PyMuPDF."""
    import pymupdf as fitz
    pages: list[str] = []
    with fitz.open(pdf_path) as document:
        for page_number, page in enumerate(document, start=1):
            try:
                pages.append(f"--- PAGE {page_number} ---\n{page.get_text('text') or ''}")
            except Exception as exc:
                LOGGER.exception("Digital extraction failed for %s page %d", pdf_path.name, page_number)
                pages.append(f"--- PAGE {page_number} ---\n[EXTRACTION ERROR: {exc}]")
    return "\n\n".join(pages)


def _configure_tesseract() -> Any:
    """Configure pytesseract and return it, with a Windows-specific diagnostic."""
    try:
        import pytesseract
    except ImportError as exc:
        raise RuntimeError("pytesseract is not installed. Run: pip install pytesseract Pillow") from exc
    candidate = TESSERACT_CMD
    if not candidate:
        standard = Path(r"C:\Program Files\Tesseract-OCR\tesseract.exe")
        candidate = str(standard) if standard.exists() else shutil.which("tesseract")
    if candidate:
        pytesseract.pytesseract.tesseract_cmd = str(candidate)
    try:
        pytesseract.get_tesseract_version()
    except Exception as exc:
        raise RuntimeError(
            "Tesseract OCR is unavailable. Install it from "
            "https://github.com/UB-Mannheim/tesseract/wiki, then add it to PATH "
            r"or set TESSERACT_CMD = r'C:\Program Files\Tesseract-OCR\tesseract.exe'."
        ) from exc
    return pytesseract


def extract_text_with_ocr(pdf_path: Path) -> str:
    """Render each PDF page locally and OCR it with Tesseract."""
    import pymupdf as fitz
    from PIL import Image, ImageEnhance, ImageOps
    pytesseract = _configure_tesseract()
    pages: list[str] = []
    matrix = fitz.Matrix(OCR_DPI / 72, OCR_DPI / 72)
    with fitz.open(pdf_path) as document:
        for page_number, page in enumerate(document, start=1):
            try:
                pixmap = page.get_pixmap(matrix=matrix, alpha=False)
                image = Image.frombytes("RGB", [pixmap.width, pixmap.height], pixmap.samples)
                image = ImageEnhance.Contrast(ImageOps.grayscale(image)).enhance(1.5)
                if OCR_THRESHOLD:
                    image = image.point(lambda value: 255 if value > 165 else 0)
                page_text = pytesseract.image_to_string(image)
                pages.append(f"--- PAGE {page_number} ---\n{page_text}")
            except Exception as exc:
                LOGGER.exception("OCR failed for %s page %d", pdf_path.name, page_number)
                pages.append(f"--- PAGE {page_number} ---\n[OCR ERROR: {exc}]")
    return "\n\n".join(pages)


def extract_tables(pdf_path: Path) -> str:
    """Best-effort digital table extraction; ordinary text extraction remains primary."""
    try:
        import pdfplumber
    except ImportError:
        return ""
    table_lines: list[str] = []
    try:
        with pdfplumber.open(pdf_path) as document:
            for page_number, page in enumerate(document.pages, start=1):
                for table in page.extract_tables() or []:
                    for row in table:
                        table_lines.append(f"{TABLE_MARKER} PAGE {page_number}: " + " | ".join(cell or "" for cell in row))
    except Exception as exc:
        LOGGER.warning("Table extraction failed for %s: %s", pdf_path.name, exc)
    return "\n".join(table_lines)


def clean_extracted_text(raw_text: str) -> str:
    """Conservatively normalize OCR/text artifacts while retaining source traceability."""
    text = unicodedata.normalize("NFKC", raw_text).replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"(?<=\d),(?=\d{3}\b)", "", text)       # 3,450 m -> 3450 m
    text = re.sub(r"(?<=\d)\s+(?=\d{3}\s*(?:m|metres?)\b)", "", text, flags=re.I)
    text = re.sub(r"\b([0-9]+)\s*[mM](?=\b)", r"\1 m", text)
    text = re.sub(r"\b(?:M[Ww]|mud\s*wt\.?|mud\s*weight)\b", "Mud Weight", text, flags=re.I)
    text = re.sub(r"\bR\s*O\s*P\b", "ROP", text, flags=re.I)
    text = re.sub(r"\bW\s*O\s*B\b", "WOB", text, flags=re.I)
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.split("\n")]
    repeated = {line for line, count in Counter(line for line in lines if len(line) > 8).items() if count >= 3}
    # Preserve page boundaries even if their neighbouring headers repeat.
    lines = [line for line in lines if line.startswith("--- PAGE") or line not in repeated]
    lines = [line for line in lines if not re.fullmatch(r"[^A-Za-z0-9]{4,}", line)]
    return "\n".join(lines).strip()


def detect_report_type(text: str, filename: str) -> str:
    """Identify DDR/WCR from both filename and report keywords."""
    source = f"{filename}\n{text[:10000]}".upper()
    ddr = len(re.findall(r"DAILY\s+DRILLING(?:\s+REPORT)?|\bDDR\b", source))
    wcr = len(re.findall(r"WELL\s+COMPLETION(?:\s+REPORT)?|\bWCR\b", source))
    return "DDR" if ddr > wcr and ddr else "WCR" if wcr else "UNKNOWN"


def identify_relevant_sections(text: str) -> dict[str, str]:
    """Lightweight NLP-style section/keyword detection without a heavy NLP dependency."""
    section_keywords = {
        "well_information": ("well information", "well name", "well id", "coordinates"),
        "drilling_parameters": ("drilling parameters", "mud weight", "rop", "rpm", "wob"),
        "formations": ("formation", "lithology", "stratigraphy"),
        "operations_events": ("operations", "events", "problems", "losses", "kick", "stuck pipe"),
        "depth": ("total depth", " td ", "depth"),
    }
    sections = {name: [] for name in section_keywords}
    # Report lines are sentence-like units here: keeping lines avoids breaking tables.
    for line in text.splitlines():
        normalized = f" {line.casefold()} "
        for name, keywords in section_keywords.items():
            if any(keyword in normalized for keyword in keywords):
                sections[name].append(line)
    return {name: "\n".join(lines[:100]) for name, lines in sections.items() if lines}


def _number(value: Optional[str]) -> Optional[float]:
    if not value:
        return None
    candidate = value.replace(",", "").replace(" ", "")
    try:
        return float(candidate)
    except ValueError:
        return None


def _first_number(text: str, patterns: list[str]) -> Optional[float]:
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.I)
        if match:
            return _number(match.group(1))
    return None


def normalize_well_id(value: Any, reference_wells: pd.DataFrame) -> Optional[str]:
    """Map formatting variants to a known well ID, otherwise retain deterministic form."""
    if value is None or pd.isna(value):
        return None
    compact = re.sub(r"[^A-Z0-9]", "", str(value).upper())
    known = reference_wells["well_id"].dropna().astype(str)
    for well_id in known:
        if re.sub(r"[^A-Z0-9]", "", well_id.upper()) == compact:
            return well_id
    match = re.fullmatch(r"(?:WELL)?W?(\d{1,6})", compact)
    return f"W{match.group(1).zfill(3)}" if match else compact or None


def next_well_id(reference_wells: pd.DataFrame) -> str:
    """Generate the next W### only when a report has no explicit ID or name match."""
    values = reference_wells["well_id"].dropna().astype(str)
    numbers = [int(match.group(1)) for value in values if (match := re.fullmatch(r"W(\d+)", value.upper()))]
    return f"W{(max(numbers, default=0) + 1):03d}"


def normalize_formation_name(value: Any, reference_formations: pd.DataFrame) -> Optional[str]:
    if value is None or pd.isna(value) or not str(value).strip():
        return None
    candidate = re.sub(r"\b(?:FORMATION|FM\.?)\b", "", str(value), flags=re.I).strip()
    names = reference_formations["formation_name"].dropna().astype(str).tolist()
    exact = {re.sub(r"[^a-z0-9]", "", name.lower()): name for name in names}
    key = re.sub(r"[^a-z0-9]", "", candidate.lower())
    if key in exact:
        return exact[key]
    try:
        from rapidfuzz import process, fuzz
        match = process.extractOne(candidate, names, scorer=fuzz.ratio, score_cutoff=88)
        return match[0] if match else candidate
    except ImportError:
        return candidate


def _stable_id(prefix: str, filename: str, payload: str) -> str:
    return f"{prefix}{hashlib.sha1(f'{filename}|{payload}'.encode()).hexdigest()[:10].upper()}"


def _formation_in_text(text: str, reference_formations: pd.DataFrame) -> Optional[str]:
    for name in reference_formations["formation_name"].dropna().astype(str):
        if re.search(rf"\b{re.escape(name)}(?:\s+(?:Formation|Fm\.?))?\b", text, re.I):
            return name
    match = re.search(r"\b([A-Za-z][A-Za-z -]{2,40})\s+(?:Formation|Fm\.?)\b", text, re.I)
    return match.group(1).strip() if match else None


def extract_with_local_rules(text: str, filename: str, references: dict[str, pd.DataFrame]) -> dict[str, list[dict[str, Any]]]:
    """API-free deterministic extraction for well IDs, depths, parameters, and events."""
    well_match = re.search(r"\b(?:well\s*(?:id|no\.?|number)?\s*[:#-]?\s*)?(W(?:ELL)?[- ]?\d{1,6})\b", text, re.I)
    well_id = normalize_well_id(well_match.group(1), references["wells"]) if well_match else None
    if not well_id:
        name_match = re.search(r"\bwell\s*name\s*[:=-]\s*([^\n]+)", text, re.I)
        if name_match:
            reported_name = name_match.group(1).strip().casefold()
            candidates = references["wells"].dropna(subset=["well_name"])
            found = candidates[candidates["well_name"].str.casefold() == reported_name]
            well_id = str(found.iloc[0]["well_id"]) if not found.empty else None
    well_id = well_id or next_well_id(references["wells"])
    depth = _first_number(text, [r"(?:total\s*depth|\bTD)\s*[:=]?\s*([\d, ]+(?:\.\d+)?)\s*m?", r"\bdepth\s*[:=]\s*([\d, ]+(?:\.\d+)?)\s*m?"])
    latitude = _first_number(text, [r"\b(?:lat(?:itude)?)\s*[:=]\s*(-?\d+(?:\.\d+)?)"])
    longitude = _first_number(text, [r"\b(?:lon(?:gitude)?)\s*[:=]\s*(-?\d+(?:\.\d+)?)"])
    formation = normalize_formation_name(_formation_in_text(text, references["formations"]), references["formations"])
    well = {"well_id": well_id, "total_depth_m": depth, "latitude": latitude, "longitude": longitude,
            "primary_formation": formation}
    result: dict[str, list[dict[str, Any]]] = {key: [] for key in DATASET_KEYS}
    result["wells"].append(well)
    if formation:
        result["formations"].append({"formation_id": _stable_id("FM", filename, formation), "formation_name": formation})
    top = _first_number(text, [r"(?:top|from)\s*[:=]?\s*([\d, ]+(?:\.\d+)?)\s*m"])
    bottom = _first_number(text, [r"(?:bottom|to)\s*[:=]?\s*([\d, ]+(?:\.\d+)?)\s*m"])
    if formation and top is not None and bottom is not None:
        result["well_formations"].append({"well_id": well_id, "formation_name": formation, "top_depth_m": top, "bottom_depth_m": bottom})
    params = {"well_id": well_id, "depth_m": depth,
              "mud_weight_ppg": _first_number(text, [r"(?:mud\s*weight|\bMW)\s*[:=]?\s*([\d.]+)\s*ppg"]),
              "rop_m_per_hr": _first_number(text, [r"\bROP\s*[:=]?\s*([\d.]+)\s*(?:m/(?:hr|h)|m/hr)?"]),
              "rpm": _first_number(text, [r"\bRPM\s*[:=]?\s*([\d.]+)"]),
              "wob_klbf": _first_number(text, [r"\bWOB\s*[:=]?\s*([\d.]+)\s*(?:klbf)?"])}
    if any(params[field] is not None for field in ("mud_weight_ppg", "rop_m_per_hr", "rpm", "wob_klbf")):
        params["record_id"] = _stable_id("PR", filename, json.dumps(params, sort_keys=True))
        result["drilling_parameters"].append(params)
    event_terms = {"lost circulation": "Mud Loss", "mud loss": "Mud Loss", "stuck pipe": "Stuck Pipe",
                   "kick": "Kick", "equipment failure": "Equipment Failure", "formation pressure": "Formation Pressure", "delay": "Delay"}
    for line in text.splitlines():
        lower = line.lower()
        event_type = next((label for term, label in event_terms.items() if term in lower), None)
        if not event_type:
            continue
        event_depth = _first_number(line, [r"(?:at|depth)\s*([\d, ]+(?:\.\d+)?)\s*m", r"\b([\d, ]{2,})\s*m\b"]) or depth
        duration = _first_number(line, [r"([\d.]+)\s*(?:hours?|hrs?)\b"])
        severity = next((word.title() for word in ("critical", "high", "medium", "low") if word in lower), None)
        event = {"well_id": well_id, "depth_m": event_depth, "formation": formation, "event_type": event_type,
                 "severity": severity, "event_description": line[:1000], "duration_hours": duration}
        event["event_id"] = _stable_id("EV", filename, json.dumps(event, sort_keys=True))
        result["drilling_events"].append(event)
    return result


def extract_with_langchain_openai(text: str, filename: str, report_type: str, references: dict[str, pd.DataFrame]) -> dict[str, list[dict[str, Any]]]:
    """Use LangChain structured output when an OpenAI key and packages are available."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OpenAI mode requires OPENAI_API_KEY in the environment or .env file.")
    try:
        from langchain_openai import ChatOpenAI
    except ImportError as exc:
        raise RuntimeError("OpenAI mode requires langchain and langchain-openai. Run pip install -r requirements.txt") from exc
    sections = identify_relevant_sections(text)
    relevant = "\n\n".join(f"[{name}]\n{value}" for name, value in sections.items())[:40000]
    prompt = ("Extract only facts supported by this petroleum report. Return null/empty lists for absent facts; never invent values. "
              f"Report type: {report_type}; filename: {filename}. Canonical well IDs include "
              f"{', '.join(references['wells']['well_id'].dropna().astype(str).head(30))}.\nRelevant report text:\n{relevant}")
    payload = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=api_key).with_structured_output(ExtractionPayload).invoke(prompt)
    return {key: [item.model_dump() for item in getattr(payload, key)] for key in DATASET_KEYS}


def extract_structured_data(text: str, filename: str, report_type: str, mode: str, references: dict[str, pd.DataFrame]) -> tuple[dict[str, list[dict[str, Any]]], str, list[str]]:
    """Choose OpenAI or local extraction; auto gracefully falls back if unavailable."""
    warnings: list[str] = []
    wants_openai = mode == "openai" or (mode == "auto" and bool(os.getenv("OPENAI_API_KEY")))
    if wants_openai:
        try:
            return extract_with_langchain_openai(text, filename, report_type, references), "OPENAI", warnings
        except Exception as exc:
            if mode == "openai":
                raise
            warnings.append(f"OpenAI unavailable; used local fallback: {exc}")
    return extract_with_local_rules(text, filename, references), "LOCAL", warnings


def validate_extracted_data(data: dict[str, list[dict[str, Any]]], filename: str) -> tuple[dict[str, list[dict[str, Any]]], list[dict[str, Any]], list[str]]:
    """Apply Pydantic and business rules, rejecting invalid records without stopping a report."""
    valid = {key: [] for key in DATASET_KEYS}
    invalid: list[dict[str, Any]] = []
    warnings: list[str] = []
    for key, records in data.items():
        model = MODEL_BY_KEY[key]
        for record in records:
            try:
                clean = model.model_validate(record).model_dump()
                if key in {"wells", "well_formations", "drilling_parameters", "drilling_events"} and not clean.get("well_id"):
                    raise ValueError("well_id must not be empty")
                if key == "well_formations" and clean["top_depth_m"] is not None and clean["bottom_depth_m"] is not None and clean["top_depth_m"] > clean["bottom_depth_m"]:
                    raise ValueError("top_depth_m must be <= bottom_depth_m")
                if key in {"formations", "well_formations"} and any(clean.values()) and not clean.get("formation_name"):
                    raise ValueError("formation_name must not be empty when formation data exists")
                valid[key].append(clean)
            except (ValidationError, ValueError) as exc:
                invalid.append({"source_filename": filename, "dataset": key, "record": record, "error": str(exc)})
    if invalid:
        warnings.append(f"Rejected {len(invalid)} invalid record(s)")
    return valid, invalid, warnings


def aggregate_data(target: dict[str, list[dict[str, Any]]], data: dict[str, list[dict[str, Any]]]) -> None:
    for key in DATASET_KEYS:
        target[key].extend(data[key])


def _normalise_frame(records: list[dict[str, Any]], columns: list[str]) -> pd.DataFrame:
    frame = pd.DataFrame(records)
    for column in columns:
        if column not in frame:
            frame[column] = pd.NA
    frame = frame[columns]
    for column in columns:
        if column.endswith(("_m", "_ppg", "_hr", "rpm", "_klbf", "latitude", "longitude", "duration_hours")):
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
        elif column.endswith("_id") or column in {"well_id", "formation_name", "well_name"}:
            frame[column] = frame[column].astype("string")
    return frame


def save_outputs(aggregated: dict[str, list[dict[str, Any]]], schemas: dict[str, list[str]], output_dir: Path) -> None:
    """Write exactly the source schema and filenames; source datasets are never written."""
    output_dir.mkdir(parents=True, exist_ok=True)
    for key, filename in DATASET_KEYS.items():
        frame = _normalise_frame(aggregated[key], schemas[key])
        subset = [col for col in ("event_id", "record_id", "well_id", "formation_name") if col in frame.columns]
        if subset:
            frame = frame.drop_duplicates(subset=subset, keep="first")
        frame.to_csv(output_dir / filename, index=False)


def process_reports(input_dir: Path, dataset_dir: Path, output_dir: Path, mode: str) -> int:
    schemas, references = load_dataset_schema(dataset_dir)
    pdf_files = ([input_dir] if input_dir.is_file() and input_dir.suffix.lower() == ".pdf"
                 else sorted(input_dir.rglob("*.pdf")) if input_dir.exists() else [])
    print(f"Found PDFs: {len(pdf_files)}")
    extracted_dir = output_dir / "extracted_text"
    extracted_dir.mkdir(parents=True, exist_ok=True)
    logs: list[dict[str, Any]] = []
    validation_report: dict[str, Any] = {"invalid_records": [], "warnings": [], "corrected_fields": []}
    aggregated = {key: [] for key in DATASET_KEYS}
    counts = Counter()
    for index, pdf_path in enumerate(pdf_files, 1):
        entry: dict[str, Any] = {"filename": pdf_path.name, "status": "FAILED", "warnings": []}
        print(f"\n[{index}/{len(pdf_files)}] {pdf_path.name}")
        try:
            pdf_type = detect_pdf_type(pdf_path)
            entry["pdf_type"] = pdf_type
            counts[pdf_type] += 1
            entry["ocr_used"] = pdf_type == "SCANNED"
            raw_text = extract_text_from_digital_pdf(pdf_path) if pdf_type == "DIGITAL" else extract_text_with_ocr(pdf_path)
            if pdf_type == "SCANNED": counts["ocr_success"] += 1
            if pdf_type == "DIGITAL": raw_text += "\n" + extract_tables(pdf_path)
            print("Text Extraction: SUCCESS")
            clean_text = clean_extracted_text(raw_text)
            (extracted_dir / f"{pdf_path.stem}.txt").write_text(clean_text, encoding="utf-8")
            print("Text Cleaning: SUCCESS")
            report_type = detect_report_type(clean_text, pdf_path.name)
            entry["report_type"] = report_type
            print(f"Report Type: {report_type}")
            data, actual_mode, warnings = extract_structured_data(clean_text, pdf_path.name, report_type, mode, references)
            entry["extraction_mode"] = actual_mode
            entry["warnings"].extend(warnings)
            print(f"AI Extraction: {actual_mode}")
            valid, invalid, validation_warnings = validate_extracted_data(data, pdf_path.name)
            entry["warnings"].extend(validation_warnings)
            validation_report["invalid_records"].extend(invalid)
            validation_report["warnings"].extend({"source_filename": pdf_path.name, "warning": item} for item in entry["warnings"])
            aggregate_data(aggregated, valid)
            entry["status"] = "SUCCESS"
            counts["success"] += 1
            print("Validation: PASS" if not invalid else "Validation: WARNINGS")
            print("Status: SUCCESS")
        except Exception as exc:
            entry["error"] = str(exc)
            counts["failed"] += 1
            print(f"Status: FAILED - {exc}")
            LOGGER.error("Failed to process %s: %s", pdf_path.name, exc)
        logs.append(entry)
    save_outputs(aggregated, schemas, output_dir)
    (output_dir / "processing_log.json").write_text(json.dumps(logs, indent=2), encoding="utf-8")
    (output_dir / "validation_report.json").write_text(json.dumps(validation_report, indent=2), encoding="utf-8")
    print("\n## Processing Summary")
    print(f"Total PDFs: {len(pdf_files)}\nDigital PDFs: {counts['DIGITAL']}\nScanned PDFs: {counts['SCANNED']}\nOCR Success: {counts['ocr_success']}\nExtraction Success: {counts['success']}\nExtraction Failed: {counts['failed']}\nValidation Warnings: {len(validation_report['warnings'])}\nOutput: {output_dir}")
    return 0 if not counts["failed"] else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=DEFAULT_INPUT_DIR, help="Directory recursively containing PDFs")
    parser.add_argument("--dataset", default=DEFAULT_DATASET_DIR, help="Synthetic dataset/schema directory")
    parser.add_argument("--output", default=DEFAULT_OUTPUT_DIR, help="Output directory")
    parser.add_argument("--mode", choices=("auto", "openai", "local"), default="auto")
    args = parser.parse_args()
    if load_dotenv:
        load_dotenv()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    return process_reports(Path(args.input), Path(args.dataset), Path(args.output), args.mode)


if __name__ == "__main__":
    raise SystemExit(main())
