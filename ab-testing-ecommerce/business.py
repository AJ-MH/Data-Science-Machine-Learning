"""
Business layer: statistics and chart building.
"""

import math

import numpy as np
import plotly.express as px
import scipy.stats
from database import SessionRepository
from statsmodels.stats.contingency_tables import Table2x2
from statsmodels.stats.power import GofChisquarePower


class GraphBuilder:
    """Builds all visualizations for the A/B test dashboard.

    Parameters
    ----------
    repo : SessionRepository
        Data source.
    """

    def __init__(self, repo=None):
        self.repo = repo or SessionRepository()

    def build_daily_sessions_chart(self):
        """Line chart of daily sessions per group over experiment window.

        Returns
        -------
        plotly Figure
        """
        daily = self.repo.get_daily_sessions()
        fig = px.line(
            daily,
            x="date",
            y="sessions",
            color="group",
            title="Daily Sessions per Group",
            color_discrete_map={"control": "#636EFA", "treatment": "#EF553B"}
        )
        fig.update_layout(xaxis_title="Date", yaxis_title="Sessions")
        return fig

    def build_conversion_bar(self):
        """Side-by-side bar chart of conversion counts per group.

        Returns
        -------
        plotly Figure
        """
        data = self.repo.get_conversion_counts(normalize=False)
        fig = px.bar(
            data,
            barmode="group",
            title="Add-to-Cart Conversions by Group",
            color_discrete_sequence=["#636EFA", "#EF553B"]
        )
        fig.update_layout(
            xaxis_title="Group",
            yaxis_title="Frequency [count]",
            legend_title="Converted"
        )
        return fig

    def build_conversion_rate_bar(self):
        """Bar chart of conversion RATES (%) per group.

        Returns
        -------
        plotly Figure
        """
        data = self.repo.get_conversion_counts(normalize=True) * 100
        fig = px.bar(
            data[["converted"]].rename(columns={"converted": "conversion_rate_pct"}),
            title="Conversion Rate by Group (%)",
            color_discrete_sequence=["#00CC96"]
        )
        fig.update_layout(xaxis_title="Group", yaxis_title="Conversion Rate [%]")
        return fig

    def build_device_chart(self):
        """Grouped bar chart of conversion rate by device and group.

        Returns
        -------
        plotly Figure
        """
        data = self.repo.get_device_conversion()
        data["conversion_rate"] = (data["conversion_rate"] * 100).round(2)
        fig = px.bar(
            data,
            x="device",
            y="conversion_rate",
            color="group",
            barmode="group",
            title="Conversion Rate by Device and Group",
            color_discrete_map={"control": "#636EFA", "treatment": "#EF553B"}
        )
        fig.update_layout(xaxis_title="Device", yaxis_title="Conversion Rate [%]")
        return fig

    def build_country_chart(self):
        """Horizontal bar chart of top 10 countries by session count.

        Returns
        -------
        plotly Figure
        """
        data = self.repo.get_country_counts().head(10)
        fig = px.bar(
            data,
            x="count",
            y="country_name",
            orientation="h",
            title="Top 10 Countries by Sessions",
            color_discrete_sequence=["#AB63FA"]
        )
        fig.update_layout(xaxis_title="Sessions", yaxis_title="Country")
        return fig


class StatsBuilder:
    """Statistical analysis for the A/B test.

    Parameters
    ----------
    repo : SessionRepository
        Data source.
    """

    def __init__(self, repo=None):
        self.repo = repo or SessionRepository()

    def calculate_n_obs(self, effect_size, alpha=0.05, power=0.8):
        """Calculate total observations needed to detect an effect size.

        Parameters
        ----------
        effect_size : float
            Cohen's w — the effect size to detect.
        alpha : float, optional
            Significance level, by default 0.05.
        power : float, optional
            Statistical power, by default 0.8.

        Returns
        -------
        int
            Total observations needed (both groups combined).
        """
        chi_power = GofChisquarePower()
        group_size = math.ceil(
            chi_power.solve_power(effect_size=effect_size, alpha=alpha, power=power)
        )
        return group_size * 2

    def get_conversion_rates(self):
        """Return conversion rate for each group.

        Returns
        -------
        dict with keys 'control' and 'treatment'
        """
        df = self.repo.get_raw()
        rates = df.groupby("group")["converted"].mean()
        return rates.to_dict()

    def get_contingency_table(self):
        """Return 2x2 contingency table as DataFrame.

        Returns
        -------
        pd.DataFrame
        """
        return self.repo.get_conversion_counts(normalize=False)

    def run_chi_square(self):
        """Run chi-square test of independence on the contingency table.

        Returns
        -------
        dict with keys: statistic, pvalue, df, odds_ratio, significant
        """
        data = self.get_contingency_table()
        table = Table2x2(data.values)
        result = table.test_nominal_association()
        odds_ratio = table.oddsratio
        return {
            "statistic": round(result.statistic, 4),
            "pvalue": round(result.pvalue, 4),
            "df": result.df,
            "odds_ratio": round(odds_ratio, 4),
            "significant": result.pvalue <= 0.05
        }

    def calculate_lift(self):
        """Calculate relative lift of treatment over control.

        Lift = (treatment_rate - control_rate) / control_rate * 100

        Returns
        -------
        float
            Lift as a percentage.
        """
        rates = self.get_conversion_rates()
        lift = (rates["treatment"] - rates["control"]) / rates["control"] * 100
        return round(lift, 2)