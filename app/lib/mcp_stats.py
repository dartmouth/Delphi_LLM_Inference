# ------------------------------------------------------------
# MCP server that lets an LLM read an uploaded Excel/CSV file and build the
# Delphi-style statistics table (same output as pages/Delphi_Analysis_Report.py).
#
# Run standalone (stdio):   python lib/mcp_stats.py
# Inspect with:             mcp dev lib/mcp_stats.py
#
# Tools are stateless: every call names a file inside UPLOAD_DIR, so the server
# can be restarted per call (as langchain-mcp-adapters does by default).
# ------------------------------------------------------------
import os
import sys
import tempfile
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent.parent
if str(APP_DIR) not in sys.path:
    sys.path.append(str(APP_DIR))

import pandas as pd
from mcp.server.fastmcp import FastMCP

from lib.score_helper import (
    DEFAULT_LIKERT_MAP,
    build_stats_table,
    compute_consensus_reached,
    compute_mean_scores,
    compute_percent_agreement,
    load_data,
    map_likert_to_numeric,
    select_question_columns,
    split_header_rows,
)

UPLOAD_DIR = Path(
    os.environ.get("MCP_UPLOAD_DIR", Path(tempfile.gettempdir()) / "agentic_uploads")
).resolve()
OUTPUT_DIR = UPLOAD_DIR / "outputs"
ALLOWED_SUFFIXES = {".csv", ".xlsx", ".xls"}
MAX_PREVIEW_ROWS = 10
MAX_COLUMNS_LISTED = 200

mcp = FastMCP("delphi-stats")


# ---------------------------------------------------------------- helpers
def _resolve(file_name: str) -> Path:
    """Resolve `file_name` inside UPLOAD_DIR, rejecting paths that escape it."""
    path = (UPLOAD_DIR / file_name).resolve()
    if UPLOAD_DIR not in path.parents:
        raise ValueError("file_name must be a file inside the upload directory.")
    if path.suffix.lower() not in ALLOWED_SUFFIXES:
        raise ValueError(f"Unsupported file type '{path.suffix}'. Use .csv, .xlsx or .xls.")
    if not path.is_file():
        raise FileNotFoundError(
            f"'{file_name}' not found. Call list_uploaded_files to see what is available."
        )
    return path


def _load(file_name: str, sheet_name: str | int = 0):
    """Read the file and split off the Group / Question ID header rows if present."""
    path = _resolve(file_name)
    df_raw = load_data(path, sheet_name=sheet_name)
    return split_header_rows(df_raw)[:2]


def _to_markdown(df: pd.DataFrame) -> str:
    cells = df.astype(str).replace({"<NA>": "", "nan": ""})
    header = "| " + " | ".join(map(str, cells.columns)) + " |"
    sep = "| " + " | ".join("---" for _ in cells.columns) + " |"
    rows = ["| " + " | ".join(r) + " |" for r in cells.itertuples(index=False)]
    return "\n".join([header, sep, *rows])


def _likert(use_default_likert_map: bool, custom_likert_map: dict | None) -> dict:
    if custom_likert_map:
        return custom_likert_map
    return DEFAULT_LIKERT_MAP if use_default_likert_map else {}


def _score(df_raw, question_cols, likert_map):
    """Numeric scores + mean per question, mirroring the page (drops all-NaN columns)."""
    cols = question_cols or select_question_columns(df_raw) or list(df_raw.columns)
    missing = [c for c in cols if c not in df_raw.columns]
    if missing:
        raise ValueError(f"Unknown columns: {missing}. Call list_question_columns first.")
    df_num = map_likert_to_numeric(df_raw[cols], likert_map)
    mean_scores = compute_mean_scores(df_num).dropna()
    if mean_scores.empty:
        raise ValueError("No numeric scores could be computed from the selected columns.")
    return df_num, mean_scores


# ------------------------------------------------------------------ tools
@mcp.tool()
def list_uploaded_files() -> list[str]:
    """List the CSV/Excel files available to analyse. Call this first."""
    if not UPLOAD_DIR.is_dir():
        return []
    return sorted(
        p.name for p in UPLOAD_DIR.iterdir() if p.suffix.lower() in ALLOWED_SUFFIXES
    )


@mcp.tool()
def inspect_file(file_name: str, sheet_name: str = "0") -> dict:
    """Read an uploaded file and return its shape, column names and first rows.

    Args:
        file_name: name of a file returned by list_uploaded_files.
        sheet_name: Excel sheet name or zero-based index (ignored for CSV). Default "0".
    """
    sheet = int(sheet_name) if str(sheet_name).isdigit() else sheet_name
    df_raw, group_map = _load(file_name, sheet)
    sheets = None
    if _resolve(file_name).suffix.lower() != ".csv":
        sheets = pd.ExcelFile(_resolve(file_name)).sheet_names
    return {
        "rows": len(df_raw),
        "columns_count": len(df_raw.columns),
        "columns": [str(c) for c in df_raw.columns][:MAX_COLUMNS_LISTED],
        "has_group_header_rows": group_map is not None,
        "sheets": sheets,
        "preview": df_raw.head(MAX_PREVIEW_ROWS).astype(str).to_dict(orient="records"),
    }


@mcp.tool()
def list_question_columns(file_name: str, sheet_name: str = "0") -> dict:
    """Detect the question columns (names starting with 'Q') and their Groups.

    Use the returned `question_columns` as `question_cols` for build_statistics_table.
    """
    sheet = int(sheet_name) if str(sheet_name).isdigit() else sheet_name
    df_raw, group_map = _load(file_name, sheet)
    cols = select_question_columns(df_raw) or list(df_raw.columns)
    return {
        "question_columns": [str(c) for c in cols][:MAX_COLUMNS_LISTED],
        "groups": (
            {str(c): str(group_map[c]) for c in cols if c in group_map.index}
            if group_map is not None
            else None
        ),
        "default_likert_map": DEFAULT_LIKERT_MAP,
    }


@mcp.tool()
def build_statistics_table(
    file_name: str,
    question_cols: list[str] | None = None,
    sheet_name: str = "0",
    use_default_likert_map: bool = True,
    custom_likert_map: dict[str, float] | None = None,
    combine_groups: list[list[str]] | None = None,
    show_count: bool = False,
) -> str:
    """Build the per-question statistics table as markdown (and save it as CSV).

    Columns: Group, Question ID, Total, Most selected ("Agree (4/10)") and the
    percentage of each answer category. Rows are sorted by mean score, highest first.

    Args:
        file_name: file from list_uploaded_files.
        question_cols: columns to score; omit to auto-detect columns starting with 'Q'.
        sheet_name: Excel sheet name or zero-based index. Default "0".
        use_default_likert_map: map Strongly disagree..Strongly agree to 1-5. If
            False, answers are parsed as numbers and only Total is reported.
        custom_likert_map: optional {answer text: number} replacing the default map.
        combine_groups: answer categories to merge into one column, e.g.
            [["Agree", "Strongly agree"], ["Disagree", "Strongly disagree"]].
            Categories not listed stay separate.
        show_count: show cells as "40.0% (4)" instead of "40.0%".
    """
    sheet = int(sheet_name) if str(sheet_name).isdigit() else sheet_name
    df_raw, group_map = _load(file_name, sheet)
    likert_map = _likert(use_default_likert_map, custom_likert_map)
    _, mean_scores = _score(df_raw, question_cols, likert_map)

    if combine_groups:
        unknown = {c for g in combine_groups for c in g} - set(likert_map)
        if unknown:
            raise ValueError(
                f"Unknown answer categories {sorted(unknown)}; valid: {list(likert_map)}."
            )
    table, _, _ = build_stats_table(
        df_raw,
        mean_scores.index.tolist(),
        likert_map,
        group_map=group_map,
        combine_groups=combine_groups,
        show_count=show_count,
    )
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = OUTPUT_DIR / f"{Path(file_name).stem}_statistics.csv"
    table.to_csv(csv_path, index=False)
    return f"{_to_markdown(table)}\n\nCSV saved to: {csv_path}"


@mcp.tool()
def compute_consensus(
    file_name: str,
    question_cols: list[str] | None = None,
    sheet_name: str = "0",
    use_default_likert_map: bool = True,
    consensus_min: float = 4,
    consensus_max: float = 5,
    threshold_percent: float = 75,
) -> str:
    """Percent agreement per question and whether consensus is reached, as markdown.

    A question reaches consensus when the share of ratings within
    [consensus_min, consensus_max] is at least threshold_percent.
    """
    sheet = int(sheet_name) if str(sheet_name).isdigit() else sheet_name
    df_raw, _ = _load(file_name, sheet)
    likert_map = _likert(use_default_likert_map, None)
    df_num, mean_scores = _score(df_raw, question_cols, likert_map)
    pct = compute_percent_agreement(df_num[mean_scores.index], consensus_min, consensus_max)
    reached = compute_consensus_reached(pct, threshold_percent)
    out = pd.DataFrame(
        {
            "Question": mean_scores.index,
            "Mean": mean_scores.round(2).values,
            "% agreement": pct.round(1).astype(str).add("%").values,
            "Consensus reached": reached.map({True: "Yes", False: "No"}).values,
        }
    )
    return _to_markdown(out)


if __name__ == "__main__":
    mcp.run()
