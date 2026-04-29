"""
Data layer: reads and shapes raw data from CSV.
"""

import os
import pandas as pd
from country_converter import CountryConverter


class SessionRepository:
    """Repository for TechCart A/B test session data.

    Parameters
    ----------
    filepath : str
        Path to the CSV file containing session data.

    Attributes
    ----------
    df : pd.DataFrame
        Raw session data loaded into memory.
    """

    def __init__(self, filepath=None):
        if filepath is None:
            base_dir = os.path.dirname(os.path.abspath(__file__))
            filepath = os.path.join(base_dir, "data", "ab_test_sessions.csv")
        self.df = pd.read_csv(filepath, parse_dates=["timestamp"])

    def get_raw(self):
        """Return the full raw DataFrame.

        Returns
        -------
        pd.DataFrame
        """
        return self.df.copy()

    def get_conversion_counts(self, normalize=False):
        """Return a 2x2 contingency table of group vs conversion.

        Parameters
        ----------
        normalize : bool, optional
            If True, return proportions instead of counts, by default False.

        Returns
        -------
        pd.DataFrame
            Rows = group (control / treatment)
            Columns = converted (0 / 1)
        """
        table = pd.crosstab(
            index=self.df["group"],
            columns=self.df["converted"],
            normalize="index" if normalize else False
        )
        table.columns = ["not_converted", "converted"]
        return table

    def get_daily_sessions(self):
        """Return number of sessions per day per group.

        Returns
        -------
        pd.DataFrame
            Columns: date, group, sessions
        """
        df = self.df.copy()
        df["date"] = df["timestamp"].dt.date
        daily = (
            df.groupby(["date", "group"])
            .size()
            .reset_index(name="sessions")
        )
        return daily

    def get_device_conversion(self):
        """Return conversion rate per device type per group.

        Returns
        -------
        pd.DataFrame
        """
        return (
            self.df.groupby(["group", "device"])["converted"]
            .mean()
            .reset_index(name="conversion_rate")
        )

    def get_country_counts(self):
        """Return session counts per country with full country names.

        Returns
        -------
        pd.DataFrame
            Columns: country_iso2, count, country_name
        """
        counts = (
            self.df["country"]
            .value_counts()
            .reset_index()
        )
        counts.columns = ["country_iso2", "count"]
        cc = CountryConverter()
        counts["country_name"] = cc.convert(counts["country_iso2"], to="name_short")
        return counts

    def get_session_stats(self):
        """Return mean session duration and pages viewed per group.

        Returns
        -------
        pd.DataFrame
        """
        return (
            self.df.groupby("group")[["session_duration_sec", "pages_viewed"]]
            .mean()
            .round(2)
            .reset_index()
        )