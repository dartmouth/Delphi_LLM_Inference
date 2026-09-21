import numpy as np
import pandas as pd
import streamlit as st
from scipy import stats

st.set_page_config(page_title="Delphi Consensus Analysis", page_icon="📊")

st.title("📊 Delphi Consensus Analysis")
st.caption(
    "Upload panelist rating data to compute the standard Delphi consensus and "
    "scale-validation metrics: central tendency, dispersion, percentage "
    "agreement, CVR/CVI, Kendall's W, and Cronbach's alpha."
)


def load_data(uploaded_file) -> pd.DataFrame:
    name = uploaded_file.name.lower()
    if name.endswith(".csv"):
        return pd.read_csv(uploaded_file)
    xls = pd.ExcelFile(uploaded_file)
    sheet = xls.sheet_names[0]
    if len(xls.sheet_names) > 1:
        sheet = st.selectbox("Sheet", xls.sheet_names)
    return xls.parse(sheet)


def kendalls_w(matrix: np.ndarray) -> dict:
    """Kendall's coefficient of concordance. matrix: (raters x items), no missing values."""
    m, n = matrix.shape
    if m < 2 or n < 2:
        return {"w": np.nan, "chi2": np.nan, "dof": np.nan, "p": np.nan}
    ranks = np.apply_along_axis(stats.rankdata, 1, matrix)
    rank_sums = ranks.sum(axis=0)
    s = np.sum((rank_sums - rank_sums.mean()) ** 2)
    tie_correction = 0.0
    for row in matrix:
        counts = np.unique(row, return_counts=True)[1]
        tie_correction += np.sum(counts**3 - counts)
    denom = m**2 * (n**3 - n) - m * tie_correction
    w = 12 * s / denom if denom else np.nan
    dof = n - 1
    if np.isnan(w):
        return {"w": w, "chi2": np.nan, "dof": dof, "p": np.nan}
    chi2 = m * dof * w
    p = stats.chi2.sf(chi2, dof)
    return {"w": w, "chi2": chi2, "dof": dof, "p": p}


def cronbachs_alpha(matrix: np.ndarray) -> float:
    """matrix: (respondents x items), no missing values."""
    k = matrix.shape[1]
    if k < 2:
        return np.nan
    item_vars = matrix.var(axis=0, ddof=1)
    total_var = matrix.sum(axis=1).var(ddof=1)
    if total_var == 0:
        return np.nan
    return (k / (k - 1)) * (1 - item_vars.sum() / total_var)


uploaded = st.file_uploader("Upload CSV or Excel file", type=["csv", "xlsx", "xls"])
if not uploaded:
    st.info("Upload a file to begin.")
    st.stop()

try:
    df = load_data(uploaded)
except Exception as e:
    st.error(f"Could not read the file: {e}")
    st.stop()

st.subheader("Preview")
st.dataframe(df.head())
st.caption(f"{len(df)} rows x {len(df.columns)} columns")

st.sidebar.header("Scale settings")
scale_min = st.sidebar.number_input("Scale minimum", value=1, step=1)
scale_max = st.sidebar.number_input("Scale maximum", value=9, step=1)
if scale_max <= scale_min:
    st.sidebar.error("Scale maximum must be greater than scale minimum.")
    st.stop()

default_lo = max(scale_min, scale_max - 2)
consensus_lo, consensus_hi = st.sidebar.slider(
    "Consensus zone",
    min_value=int(scale_min),
    max_value=int(scale_max),
    value=(int(default_lo), int(scale_max)),
    help=(
        "Ratings in this range count toward percentage agreement and "
        "CVR/CVI. Leave the upper bound at the scale maximum to treat the "
        "lower bound as an 'essential/relevant' cutoff for CVR/CVI."
    ),
)
agreement_threshold = st.sidebar.slider(
    "Consensus-reached threshold (% agreement)",
    0,
    100,
    75,
    help=(
        "An item is flagged as having reached consensus when its percentage "
        "agreement meets or exceeds this value."
    ),
)

st.subheader("Select columns")
columns = list(df.columns)
id_col = st.selectbox("Panelist ID column (optional)", ["(none)"] + columns)
candidates = [c for c in columns if c != id_col]
numeric_guess = [
    c for c in candidates if pd.to_numeric(df[c], errors="coerce").notna().any()
]
item_cols = st.multiselect("Item rating columns", candidates, default=numeric_guess)

if len(item_cols) < 1:
    st.info("Select at least one item column to compute statistics.")
    st.stop()

ratings = df[item_cols].apply(pd.to_numeric, errors="coerce")
non_numeric_count = int(df[item_cols].notna().sum().sum() - ratings.notna().sum().sum())
if non_numeric_count > 0:
    st.warning(
        f"{non_numeric_count} value(s) across the selected columns were "
        "non-numeric and treated as missing."
    )
out_of_range = int(((ratings < scale_min) | (ratings > scale_max)).sum().sum())
if out_of_range > 0:
    st.warning(
        f"{out_of_range} value(s) fall outside the {scale_min}-{scale_max} "
        "scale range."
    )

st.subheader("Per-item statistics")
rows = []
for col in item_cols:
    series = ratings[col].dropna()
    n = len(series)
    if n == 0:
        rows.append({"Item": col, "N": 0})
        continue
    q1, q3 = np.percentile(series, [25, 75])
    ne = int(((series >= consensus_lo) & (series <= consensus_hi)).sum())
    pct_agreement = 100 * ne / n
    i_cvi = ne / n
    cvr = (ne - n / 2) / (n / 2)
    rows.append(
        {
            "Item": col,
            "N": n,
            "Mean": series.mean(),
            "Median": series.median(),
            "SD": series.std(ddof=1) if n > 1 else np.nan,
            "IQR": q3 - q1,
            "% Agreement": pct_agreement,
            "I-CVI": i_cvi,
            "CVR": cvr,
            "Consensus reached": pct_agreement >= agreement_threshold,
        }
    )

item_stats = pd.DataFrame(rows).set_index("Item")
st.dataframe(
    item_stats.style.format(
        {
            "Mean": "{:.2f}",
            "Median": "{:.2f}",
            "SD": "{:.2f}",
            "IQR": "{:.2f}",
            "% Agreement": "{:.1f}%",
            "I-CVI": "{:.2f}",
            "CVR": "{:.2f}",
        }
    )
)
st.caption(
    "% Agreement and I-CVI/CVR are based on the consensus zone set in the "
    "sidebar. CVR follows Lawshe (1975): (ne - N/2) / (N/2), where ne is the "
    "number of panelists rating the item within the consensus zone. Compare "
    "CVR values against Lawshe/Ayre & Scannell critical-value tables for your "
    "panel size to test significance — this app does not embed those tables."
)

st.download_button(
    "Download per-item statistics (CSV)",
    item_stats.to_csv().encode("utf-8"),
    file_name="delphi_item_statistics.csv",
    mime="text/csv",
)

st.subheader("Scale-level statistics")
complete = ratings.dropna()
dropped = len(ratings) - len(complete)
if dropped:
    st.caption(
        f"{dropped} panelist row(s) with missing values in the selected items "
        "were excluded from Kendall's W and Cronbach's alpha (listwise "
        "deletion)."
    )

col1, col2, col3 = st.columns(3)

enough_data = len(item_cols) >= 2 and len(complete) >= 2

if enough_data:
    w_result = kendalls_w(complete.to_numpy())
    if np.isnan(w_result["w"]):
        col1.metric("Kendall's W", "n/a")
    else:
        col1.metric("Kendall's W", f"{w_result['w']:.3f}")
        col1.caption(
            f"chi²({w_result['dof']}) = {w_result['chi2']:.2f}, "
            f"p = {w_result['p']:.4f}"
        )

    alpha = cronbachs_alpha(complete.to_numpy())
    col2.metric("Cronbach's alpha", "n/a" if np.isnan(alpha) else f"{alpha:.3f}")
else:
    col1.metric("Kendall's W", "n/a")
    col1.caption("Needs ≥2 items and ≥2 complete panelist rows.")
    col2.metric("Cronbach's alpha", "n/a")
    col2.caption("Needs ≥2 items and ≥2 complete panelist rows.")

s_cvi = item_stats["I-CVI"].mean() if "I-CVI" in item_stats else np.nan
col3.metric("S-CVI/Ave", "n/a" if pd.isna(s_cvi) else f"{s_cvi:.3f}")
col3.caption("Average of item-level CVI (I-CVI) across all selected items.")

with st.expander("Interpretation notes"):
    st.markdown(
        "- **Cronbach's alpha**: <0.5 unacceptable, 0.5-0.6 poor, 0.6-0.7 "
        "questionable, 0.7-0.8 acceptable, 0.8-0.9 good, >0.9 excellent.\n"
        "- **Kendall's W**: ranges from 0 (no agreement) to 1 (complete "
        "agreement) across panelists' ratings; the chi-square test indicates "
        "whether the agreement is significantly greater than chance.\n"
        "- **S-CVI/Ave ≥ 0.90** and **I-CVI ≥ 0.78** are commonly cited "
        "acceptability benchmarks (Polit & Beck, 2006).\n"
        "- IQR is computed with linear interpolation (numpy default); some "
        "software (e.g., SPSS) uses a different quartile method and may "
        "produce slightly different values."
    )
