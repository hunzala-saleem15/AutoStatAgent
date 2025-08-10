import pandas as pd
import numpy as np
from scipy.stats import (
    ttest_ind, f_oneway, chi2_contingency, pearsonr, spearmanr,
    shapiro, mannwhitneyu, kruskal
)

class HypothesisTesting:
    @staticmethod
    def _hypothesis_text(test_type, col1, col2=None):
        if test_type == "t-test":
            return (f"H₀: The mean of '{col1}' is equal across the two groups in '{col2}'.",
                    f"H₁: The mean of '{col1}' is different across the two groups in '{col2}'.")
        elif test_type == "anova":
            return (f"H₀: The mean of '{col1}' is equal across all groups in '{col2}'.",
                    f"H₁: At least one group mean of '{col1}' in '{col2}' is different.")
        elif test_type == "mann-whitney":
            return (f"H₀: The distribution of '{col1}' is the same across the two groups in '{col2}'.",
                    f"H₁: The distribution of '{col1}' differs between the two groups in '{col2}'.")
        elif test_type == "kruskal":
            return (f"H₀: The distribution of '{col1}' is the same across all groups in '{col2}'.",
                    f"H₁: At least one group distribution of '{col1}' in '{col2}' is different.")
        elif test_type == "chi-square":
            return (f"H₀: '{col1}' and '{col2}' are independent.",
                    f"H₁: '{col1}' and '{col2}' are not independent.")
        elif test_type == "correlation":
            return (f"H₀: There is no correlation between '{col1}' and '{col2}'.",
                    f"H₁: There is a correlation between '{col1}' and '{col2}'.")
        return ("", "")

    # ---------- Effect Size Calculations ----------
    @staticmethod
    def _cohens_d(group1, group2):
        diff = group1.mean() - group2.mean()
        pooled_std = np.sqrt(((group1.std() ** 2) + (group2.std() ** 2)) / 2)
        return diff / pooled_std if pooled_std != 0 else np.nan

    @staticmethod
    def _eta_squared_anova(stat, df_between, df_within):
        return stat * df_between / (stat * df_between + df_within)

    @staticmethod
    def _cramers_v(confusion_matrix):
        chi2 = chi2_contingency(confusion_matrix)[0]
        n = confusion_matrix.sum().sum()
        phi2 = chi2 / n
        r, k = confusion_matrix.shape
        return np.sqrt(phi2 / min(k - 1, r - 1))

    # ---------- Test Functions ----------
    @staticmethod
    def _is_normal(series):
        if len(series) < 3:
            return False
        p = shapiro(series.sample(min(500, len(series))))[1]
        return p > 0.05

    @staticmethod
    def test_numeric_categorical(df, cat_col, num_col, alpha=0.05):
        groups = [group[num_col].dropna() for _, group in df.groupby(cat_col, observed=True)]
        if len(groups) < 2:
            return "Not enough groups for hypothesis testing."

        if len(groups) == 2:
            if all(HypothesisTesting._is_normal(g) for g in groups):
                stat, p = ttest_ind(groups[0], groups[1], nan_policy='omit')
                effect = HypothesisTesting._cohens_d(groups[0], groups[1])
                test_type = "t-test"
            else:
                stat, p = mannwhitneyu(groups[0], groups[1], alternative="two-sided")
                effect = None
                test_type = "mann-whitney"

        else:
            if all(HypothesisTesting._is_normal(g) for g in groups):
                stat, p = f_oneway(*groups)
                effect = HypothesisTesting._eta_squared_anova(stat, len(groups)-1, len(df)-len(groups))
                test_type = "anova"
            else:
                stat, p = kruskal(*groups)
                effect = None
                test_type = "kruskal"

        h0, h1 = HypothesisTesting._hypothesis_text(test_type, num_col, cat_col)
        return {
            "test": test_type.upper(),
            "columns": (num_col, cat_col),
            "H0": h0,
            "H1": h1,
            "stat": stat,
            "p_value": p,
            "alpha": alpha,
            "significance": "Significant" if p < alpha else "Not significant",
            "effect_size": effect
        }

    @staticmethod
    def test_categorical_categorical(df, cat1, cat2, alpha=0.05):
        contingency_table = pd.crosstab(df[cat1], df[cat2])
        if contingency_table.empty:
            return "Contingency table is empty or invalid."
        stat, p, _, _ = chi2_contingency(contingency_table)
        effect = HypothesisTesting._cramers_v(contingency_table)
        h0, h1 = HypothesisTesting._hypothesis_text("chi-square", cat1, cat2)
        return {
            "test": "Chi-square",
            "columns": (cat1, cat2),
            "H0": h0,
            "H1": h1,
            "stat": stat,
            "p_value": p,
            "alpha": alpha,
            "significance": "Significant" if p < alpha else "Not significant",
            "effect_size_cramers_v": effect
        }

    @staticmethod
    def test_numeric_numeric(df, col1, col2, alpha=0.05):
        series1, series2 = df[col1].dropna(), df[col2].dropna()
        if HypothesisTesting._is_normal(series1) and HypothesisTesting._is_normal(series2):
            stat, p = pearsonr(series1, series2)
            method = "Pearson"
        else:
            stat, p = spearmanr(series1, series2)
            method = "Spearman"

        h0, h1 = HypothesisTesting._hypothesis_text("correlation", col1, col2)
        return {
            "test": f"{method} Correlation",
            "columns": (col1, col2),
            "H0": h0,
            "H1": h1,
            "stat": stat,
            "p_value": p,
            "alpha": alpha,
            "significance": "Significant" if p < alpha else "Not significant",
            "correlation_strength": "Strong" if abs(stat) > 0.7 else "Moderate" if abs(stat) > 0.3 else "Weak"
        }
