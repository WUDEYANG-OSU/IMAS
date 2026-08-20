from __future__ import annotations
import re


def safe_name(value: object, max_length: int = 120) -> str:
    return re.sub(r"[^A-Za-z0-9_\-.]+", "_", str(value))[:max_length]


def natural_sort_key(value: object):
    parts = re.split(r"(\d+)", str(value))
    return [int(part) if part.isdigit() else part.lower() for part in parts]


def normalize_cluster(value: object) -> str:
    text = str(value).strip()
    match = re.fullmatch(r"(?:C|Cluster)[-_ ]?(\d+)", text, flags=re.I)
    if match:
        return str(int(match.group(1)))
    if re.fullmatch(r"\d+\.0+", text):
        return str(int(float(text)))
    return str(int(text)) if text.isdigit() else text


def normalize_cell_id(value: object) -> str:
    text = str(value)
    return text.split("::", 1)[-1] if "::" in text else text


def make_cell_key(sample_id: object, cell_id: object) -> str:
    return f"{str(sample_id).strip()}::{normalize_cell_id(cell_id)}"
