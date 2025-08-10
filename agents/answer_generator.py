import re
import pandas as pd
from agents.hypothesis_testing import HypothesisTesting


class AnswerGenerator:
    @staticmethod
    def format_test_result(result):
        """Format HypothesisTesting output into readable text."""
        if not isinstance(result, dict):
            return str(result)

        h0 = result.get("H0", "N/A")
        h1 = result.get("H1", "N/A")
        test_name = result.get("test", "N/A")
        conclusion = result.get("result", "N/A")

        return (
            f"Test: {test_name}\n"
            f"H₀: {h0}\n"
            f"H₁: {h1}\n"
            f"Conclusion: {conclusion}"
        )

    @staticmethod
    def extract_columns_from_question(q):
        """Extract columns mentioned in single quotes from a question."""
        return re.findall(r"'([^']+)'", q)

    @staticmethod
    def answer_questions(df, questions, alpha=0.05):
        answers = {}

        # Skip overly simple metrics
        skip_patterns = [
            r"\bmean\b", r"\baverage\b", r"\bmax\b", r"\bmin\b",
            r"\bmedian\b", r"\bsum\b", r"\bcount\b"
        ]

        for q in questions:
            try:
                q_lower = q.lower()
                if any(re.search(p, q_lower) for p in skip_patterns):
                    continue

                cols = AnswerGenerator.extract_columns_from_question(q)

                # === Correlation (numeric-numeric) ===
                if "correlation" in q_lower and len(cols) == 2:
                    col1, col2 = cols
                    if all(pd.api.types.is_numeric_dtype(df[c]) for c in cols):
                        result = HypothesisTesting.test_numeric_numeric(df, col1, col2, alpha=alpha)
                        answers[q] = AnswerGenerator.format_test_result(result)

                # === Skewness ===
                elif "skew" in q_lower and len(cols) == 1:
                    col = cols[0]
                    if pd.api.types.is_numeric_dtype(df[col]):
                        skew_val = df[col].skew()
                        skew_type = "Highly skewed" if abs(skew_val) > 1 else "Relatively symmetric"
                        answers[q] = f"'{col}' skewness = {skew_val:.2f} ({skew_type})."

                # === Kurtosis ===
                elif "kurtosis" in q_lower and len(cols) == 1:
                    col = cols[0]
                    if pd.api.types.is_numeric_dtype(df[col]):
                        kurt_val = df[col].kurtosis()
                        if kurt_val > 3:
                            shape = "Leptokurtic (heavy tails)"
                        elif kurt_val < 3:
                            shape = "Platykurtic (light tails)"
                        else:
                            shape = "Mesokurtic (normal-like)"
                        answers[q] = f"'{col}' kurtosis = {kurt_val:.2f} ({shape})."

                # === Outlier detection ===
                elif "outlier" in q_lower and len(cols) == 1:
                    col = cols[0]
                    if pd.api.types.is_numeric_dtype(df[col]):
                        Q1, Q3 = df[col].quantile([0.25, 0.75])
                        IQR = Q3 - Q1
                        outliers = ((df[col] < Q1 - 1.5 * IQR) | (df[col] > Q3 + 1.5 * IQR)).sum()
                        answers[q] = f"'{col}' has {outliers} outliers detected using IQR method."

                # === Numeric-Categorical difference (t-test / ANOVA) ===
                elif "different" in q_lower and len(cols) == 2:
                    cat_col, num_col = cols
                    if pd.api.types.is_numeric_dtype(df[num_col]) and (
                        pd.api.types.is_categorical_dtype(df[cat_col]) or df[cat_col].dtype == object
                    ):
                        result = HypothesisTesting.test_numeric_categorical(df, cat_col, num_col, alpha=alpha)
                        answers[q] = AnswerGenerator.format_test_result(result)

                # === Categorical-Categorical association (Chi-square) ===
                elif "association" in q_lower and len(cols) == 2:
                    cat1, cat2 = cols
                    if (df[cat1].dtype == object or pd.api.types.is_categorical_dtype(df[cat1])) and \
                       (df[cat2].dtype == object or pd.api.types.is_categorical_dtype(df[cat2])):
                        result = HypothesisTesting.test_categorical_categorical(df, cat1, cat2, alpha=alpha)
                        answers[q] = AnswerGenerator.format_test_result(result)

                # === Missing data check ===
                elif "missing" in q_lower and len(cols) == 1:
                    col = cols[0]
                    missing_pct = df[col].isnull().mean() * 100
                    if missing_pct > 0:
                        answers[q] = f"Missing data in '{col}': {missing_pct:.1f}%."
                    else:
                        answers[q] = f"No missing data in '{col}'."

                # === Duplicate check ===
                elif "duplicate" in q_lower:
                    duplicates = df.duplicated().sum()
                    answers[q] = f"Dataset contains {duplicates} duplicate rows."

                # === Trend detection placeholder ===
                elif "trend" in q_lower and len(cols) == 1:
                    col = cols[0]
                    answers[q] = f"Trend analysis for '{col}' requires time-series decomposition — not automated here."

            except Exception as e:
                answers[q] = f"Error answering question: {e}"

        return answers
