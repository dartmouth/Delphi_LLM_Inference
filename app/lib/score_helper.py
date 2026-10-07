# ------------------------------------------------------------
# Helper functions for scoring Likert-scale survey responses and
# plotting the mean agreement per question.
# ------------------------------------------------------------
import re

import matplotlib.figure
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import seaborn as sns
from scipy.stats import gaussian_kde

DEFAULT_LIKERT_MAP = {
    "Strongly disagree": 1,
    "Disagree": 2,
    "Neutral": 3,
    "Agree": 4,
    "Strongly agree": 5,
}


def load_data(file, sheet_name=0) -> pd.DataFrame:
    """Read an uploaded (or path-like) CSV or Excel file into a DataFrame."""
    name = getattr(file, "name", str(file)).lower()
    if name.endswith(".csv"):
        return pd.read_csv(file)
    return pd.read_excel(file, sheet_name=sheet_name)


def select_question_columns(df: pd.DataFrame, prefix: str = "Q") -> list:
    """Return the columns whose name starts with the given prefix (default 'Q')."""
    return [
        c
        for c in df.columns
        if isinstance(c, str) and c.startswith(prefix) and c != "Question ID"
    ]


def split_header_rows(df: pd.DataFrame):
    """Handle sheets whose header row holds Group names, with a 'Question ID' row
    and a question-text row beneath it, before the responses begin.

    Returns (data, groups, question_ids). `data` has the Question IDs as column
    names; `groups` and `question_ids` map those columns to their Group and ID.
    If the layout isn't detected, returns (df, None, None) unchanged.
    """
    if df.shape[0] < 2 or df.shape[1] < 2:
        return df, None, None
    first_rows = df.iloc[0, :2].astype(str).str.strip().str.lower()
    if "question id" not in first_rows.values:
        return df, None, None
    # Blank ID cells (e.g. the index column) fall back to the original header so
    # column names stay real strings (NaN names break Streamlit's JSON preview).
    ids = pd.Series(
        [
            str(v).strip() if pd.notna(v) and str(v).strip() else str(c)
            for v, c in zip(df.iloc[0], df.columns)
        ],
        index=df.columns,
    )
    # pandas renames repeated headers to 'Name.1', 'Name.2', ...
    groups = pd.Series(
        [re.sub(r"\.\d+$", "", str(c)) for c in df.columns], index=df.columns
    )
    data = df.iloc[2:].reset_index(drop=True)
    data.columns = ids.values
    data = data.loc[:, ~data.columns.duplicated()]
    groups.index = ids.values
    groups = groups[~groups.index.duplicated()]
    return data, groups, pd.Series(ids.values, index=ids.values)[data.columns]


def parse_question_id(column: str) -> tuple:
    """Split a column name such as 'Q2.3 Recording access' into (group, question_id),
    e.g. ('Q2', 'Q2.3'). Without a sub-number the id doubles as the group."""
    m = re.match(r"\s*(Q\s*\d+(?:[._-]\d+)*)", str(column), flags=re.I)
    if not m:
        return ("", "")
    qid = re.sub(r"\s+", "", m.group(1))
    return (re.split(r"[._-]", qid, maxsplit=1)[0], qid)


def map_likert_to_numeric(df: pd.DataFrame, likert_map: dict = None) -> pd.DataFrame:
    """Strip whitespace from text answers, map them to a numeric Likert scale,
    and coerce anything left (including already-numeric data) to numbers."""
    likert_map = likert_map or {}
    cleaned = df.apply(lambda col: col.map(lambda x: x.strip() if isinstance(x, str) else x))
    mapped = cleaned.replace(likert_map) if likert_map else cleaned
    return mapped.apply(pd.to_numeric, errors="coerce")


def compute_mean_scores(df_num: pd.DataFrame) -> pd.Series:
    """Mean score per question, sorted highest to lowest."""
    return df_num.mean().sort_values(ascending=False)


def compute_mean_scores_excluding_neutral(df_num: pd.DataFrame, neutral_value: float = 3) -> pd.Series:
    """Mean score per question, ignoring neutral responses, sorted highest to lowest.

    Neutral cells are masked to NaN for this calculation only; df_num itself
    is left unchanged. A question with all-neutral responses yields NaN.
    """
    return df_num.where(df_num != neutral_value).mean().sort_values(ascending=False)


def compute_answer_counts(df: pd.DataFrame, categories: list) -> pd.DataFrame:
    """Count of raw answers per category, for each column (question).

    Whitespace is stripped from text answers first, matching
    `map_likert_to_numeric`'s cleaning step. Categories with no matching
    answers for a question are filled with 0.
    """
    cleaned = df.apply(lambda col: col.map(lambda x: x.strip() if isinstance(x, str) else x))
    counts = cleaned.apply(lambda col: col.value_counts().reindex(categories, fill_value=0))
    return counts.astype(int).T


def compute_mode_answer(answer_counts: pd.DataFrame) -> pd.Series:
    """Most-selected category per question, formatted as 'Category (count/total)'.

    `answer_counts` is the per-question x per-category count table returned by
    `compute_answer_counts`. Ties keep the first category in column order.
    """
    totals = answer_counts.sum(axis=1)
    top_category = answer_counts.idxmax(axis=1)
    top_count = answer_counts.max(axis=1)
    return top_category + " (" + top_count.astype(str) + "/" + totals.astype(str) + ")"


def compute_percent_agreement(
    df_num: pd.DataFrame, consensus_lo: float, consensus_hi: float
) -> pd.Series:
    """Share (0-100) of non-missing responses per question inside [consensus_lo, consensus_hi]."""
    in_zone = df_num.apply(lambda col: col.between(consensus_lo, consensus_hi))
    return 100 * in_zone.sum() / df_num.notna().sum()


def compute_consensus_reached(pct_agreement: pd.Series, threshold: float) -> pd.Series:
    """Boolean per question: whether percent agreement meets the consensus threshold."""
    return pct_agreement >= threshold


def plot_question_ridgeline(
    df_num: pd.DataFrame,
    consensus_reached: pd.Series,
    order: list = None,
    scale_min: float = 1,
    scale_max: float = 5,
    bw_adjust: float = 1.0,
    title: str = "Rating Distribution by Question",
    color_reached: str = "#0ca30c",
    color_not_reached: str = "#d03b3b",
) -> matplotlib.figure.Figure:
    """Ridgeline (joyplot) of the response distribution for every question.

    Each question gets a stacked, KDE-smoothed density of its raw ratings,
    filled in `color_reached`/`color_not_reached` depending on
    `consensus_reached`, with a marker at the question's mean rating.
    """
    order = list(order) if order is not None else list(df_num.columns)
    x = np.linspace(scale_min, scale_max, 200)

    fig = matplotlib.figure.Figure(figsize=(7, max(3, 0.6 * len(order) + 1)))
    ax = fig.add_subplot(111)

    for i, col in enumerate(order):
        values = df_num[col].dropna().to_numpy()
        reached = bool(consensus_reached.get(col, False))
        color = color_reached if reached else color_not_reached
        if len(values) == 0:
            continue
        mean_val = values.mean()
        if np.ptp(values) == 0:
            # No variance: draw a spike instead of a degenerate KDE.
            ax.plot([mean_val, mean_val], [i, i + 1], color=color, alpha=0.8, linewidth=2)
        else:
            kde = gaussian_kde(values)
            kde.set_bandwidth(kde.factor * bw_adjust)
            y = kde(x)
            y_shifted = y / y.max() + i
            ax.fill_between(x, i, y_shifted, color=color, alpha=0.8)
        ax.plot(mean_val, i + 0.5, "o", mfc="white", color=color, mew=2)

    ax.set_yticks([i + 0.5 for i in range(len(order))])
    ax.set_yticklabels(order)
    ax.set_xlim(scale_min, scale_max)
    ax.set_xlabel(f"Rating ({scale_min:g}-{scale_max:g})")
    ax.set_title(title)
    sns.despine(ax=ax, left=True)
    fig.tight_layout()
    return fig


def plot_answer_percentages(
    percents: pd.DataFrame,
    counts: pd.DataFrame,
    title: str = "Consensus on Statements",
) -> go.Figure:
    """Horizontal stacked bar chart of answer percentages per question, annotated with counts."""
    fig = go.Figure()
    for col in percents.columns:
        pct = percents[col].fillna(0).astype(float)
        fig.add_trace(
            go.Bar(
                x=pct,
                y=percents.index,
                orientation="h",
                name=str(col),
                text=[f"{p:.0f}% ({c})" if p >= 5 else "" for p, c in zip(pct, counts[col])],
                textposition="inside",
            )
        )
    fig.update_layout(
        title=title,
        barmode="stack",
        xaxis_title="Share of answers (%)",
        yaxis_title="",
        xaxis_range=[0, 100],
    )
    fig.update_yaxes(autorange="reversed")
    return fig
