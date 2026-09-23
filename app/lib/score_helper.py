# ------------------------------------------------------------
# Helper functions for scoring Likert-scale survey responses and
# plotting the mean agreement per question.
# ------------------------------------------------------------
import pandas as pd
import plotly.graph_objects as go

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
