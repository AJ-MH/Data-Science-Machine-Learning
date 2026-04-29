"""
Presentation layer: Streamlit dashboard for TechCart A/B test results.
"""

import streamlit as st
from business import GraphBuilder, StatsBuilder

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="TechCart A/B Test Dashboard",
    page_icon="🛒",
    layout="wide"
)

st.title("🛒 TechCart — Product Page A/B Test Dashboard")
st.markdown(
    """
    **Experiment question:** Does the redesigned product page (Version B) increase
    the Add-to-Cart rate compared to the original page (Version A)?

    > Data covers 4,000 user sessions over a 14-day experiment window.
    """
)

# ── Instantiate business layer ─────────────────────────────────────────────────
gb = GraphBuilder()
sb = StatsBuilder()

# ── Section 1: Experiment Overview ────────────────────────────────────────────
st.header("1. Experiment Overview")

rates = sb.get_conversion_rates()
lift = sb.calculate_lift()
n_obs_needed = sb.calculate_n_obs(effect_size=0.2)

col1, col2, col3, col4 = st.columns(4)
col1.metric("Control Conversion Rate", f"{rates['control']*100:.2f}%")
col2.metric("Treatment Conversion Rate", f"{rates['treatment']*100:.2f}%")
col3.metric("Relative Lift", f"{lift}%", delta=f"{lift}%")
col4.metric("Obs. Needed (effect=0.2)", f"{n_obs_needed:,}")

st.plotly_chart(gb.build_daily_sessions_chart(), use_container_width=True)

# ── Section 2: Conversion Results ─────────────────────────────────────────────
st.header("2. Conversion Results")

col_left, col_right = st.columns(2)
with col_left:
    st.plotly_chart(gb.build_conversion_bar(), use_container_width=True)
with col_right:
    st.plotly_chart(gb.build_conversion_rate_bar(), use_container_width=True)

# ── Section 3: Statistical Test ───────────────────────────────────────────────
st.header("3. Chi-Square Test of Independence")

st.markdown(
    """
    We use a **chi-square test** to determine whether the difference in conversion
    rates between the two groups is statistically significant or just random variation.

    - **Null hypothesis (H₀):** There is no relationship between page version and conversion rate.
    - **Alternate hypothesis (H₁):** The page version affects conversion rate.
    - **Significance threshold:** α = 0.05
    """
)

effect_size = st.slider(
    "Effect size to detect (Cohen's w):",
    min_value=0.1,
    max_value=0.8,
    step=0.1,
    value=0.2
)
n_needed = sb.calculate_n_obs(effect_size=effect_size)
st.info(f"To detect an effect size of **{effect_size}**, you need **{n_needed:,}** total observations.")

chi_result = sb.run_chi_square()

col_a, col_b, col_c = st.columns(3)
col_a.metric("Chi-Square Statistic", chi_result["statistic"])
col_b.metric("p-value", chi_result["pvalue"])
col_c.metric("Odds Ratio", chi_result["odds_ratio"])

if chi_result["significant"]:
    st.success(
        f"✅ **Result is statistically significant** (p = {chi_result['pvalue']} ≤ 0.05). "
        f"We reject the null hypothesis. The new product page improves conversion rate."
    )
else:
    st.warning(
        f"⚠️ **Result is NOT statistically significant** (p = {chi_result['pvalue']} > 0.05). "
        f"We fail to reject the null hypothesis."
    )

contingency = sb.get_contingency_table()
st.subheader("Contingency Table")
st.dataframe(contingency, use_container_width=True)

# ── Section 4: Segment Analysis ───────────────────────────────────────────────
st.header("4. Segment Analysis")

col_dev, col_country = st.columns(2)
with col_dev:
    st.plotly_chart(gb.build_device_chart(), use_container_width=True)
with col_country:
    st.plotly_chart(gb.build_country_chart(), use_container_width=True)

st.caption("Dashboard built with Streamlit · Data: Synthetic (generated for portfolio)")