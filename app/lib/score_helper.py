# ------------------------------------------------------------
# Helper functions for scoring Likert-scale survey responses and
# plotting the mean agreement per question.
# ------------------------------------------------------------
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
    return [c for c in df.columns if isinstance(c, str) and c.startswith(prefix)]


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


def plot_mean_scores(
    mean_scores: pd.Series,
    scale_min: float = 1,
    scale_max: float = 5,
    title: str = "Consensus on Statements",
) -> go.Figure:
    """Horizontal bar chart of mean Likert scores, annotated with the exact value."""
    fig = go.Figure(
        go.Bar(
            x=mean_scores.values,
            y=mean_scores.index,
            orientation="h",
            marker_color="#2a78d6",
            text=[f"{v:.2f}" for v in mean_scores.values],
            textposition="outside",
        )
    )
    fig.update_layout(
        title=title,
        xaxis_title=f"Mean Likert Score ({scale_min:g} = lowest, {scale_max:g} = highest)",
        yaxis_title="",
        xaxis_range=[scale_min, scale_max * 1.1],
    )
    fig.update_yaxes(autorange="reversed")
    return fig
