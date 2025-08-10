import numpy as np
import pandas as pd
from scipy.stats import shapiro

class QuestionGenerator:
    @staticmethod
    def generate_questions(df, max_corr_questions=5):
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        datetime_cols = df.select_dtypes(include=['datetime', 'datetimetz']).columns.tolist()

        questions = []

        # --- Numeric-Numeric Relationships (smart correlation method selection) ---
        corr_questions_added = 0
        for i in range(len(numeric_cols)):
            for j in range(i+1, len(numeric_cols)):
                col1, col2 = numeric_cols[i], numeric_cols[j]
                if df[col1].dropna().empty or df[col2].dropna().empty:
                    continue

                # Check normality to decide Pearson vs Spearman
                p1 = shapiro(df[col1].dropna().sample(min(500, len(df[col1].dropna()))))[1] if len(df[col1].dropna()) > 3 else 1
                p2 = shapiro(df[col2].dropna().sample(min(500, len(df[col2].dropna()))))[1] if len(df[col2].dropna()) > 3 else 1
                method = "Pearson" if (p1 > 0.05 and p2 > 0.05) else "Spearman"

                corr = df[col1].corr(df[col2], method=method.lower())
                if pd.notnull(corr) and abs(corr) > 0.3 and corr_questions_added < max_corr_questions:
                    strength = "strong" if abs(corr) > 0.7 else "moderate"
                    questions.append(
                        f"Does the {strength} {method} correlation (r={corr:.2f}) between '{col1}' and '{col2}' indicate potential causal or confounding factors worth testing?"
                    )
                    corr_questions_added += 1

        # --- Distribution-related Questions ---
        for col in numeric_cols:
            skew_val = df[col].skew()
            kurt_val = df[col].kurtosis()
            if abs(skew_val) > 1:
                questions.append(f"'{col}' is highly skewed (skew={skew_val:.2f}); should transformation or robust statistics be used?")
            if kurt_val > 3:
                questions.append(f"'{col}' has high kurtosis ({kurt_val:.2f}); are heavy tails affecting hypothesis test validity?")

        # --- Categorical-Numeric Analysis (with test recommendation) ---
        for cat in cat_cols:
            if df[cat].nunique() <= 10:
                for num in numeric_cols:
                    questions.append(
                        f"Do different '{cat}' categories show significant differences in '{num}' means (t-test/ANOVA) or medians (Kruskal-Wallis) depending on normality?"
                    )

        # --- Categorical-Categorical Associations ---
        for i in range(len(cat_cols)):
            for j in range(i+1, len(cat_cols)):
                if df[cat_cols[i]].nunique() <= 10 and df[cat_cols[j]].nunique() <= 10:
                    questions.append(
                        f"Is there an association between '{cat_cols[i]}' and '{cat_cols[j]}' (Chi-square test with Cramér’s V effect size)?"
                    )

        # --- Time Series / Trends ---
        for col in datetime_cols:
            questions.append(f"Are there seasonal or trend components in '{col}' detectable via time-series decomposition?")
            for num in numeric_cols:
                questions.append(
                    f"Does '{num}' change significantly over time based on '{col}' (trend analysis, Mann-Kendall test)?"
                )

        # --- Outlier Impact ---
        for col in numeric_cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            outliers = ((df[col] < Q1 - 1.5*IQR) | (df[col] > Q3 + 1.5*IQR)).sum()
            if outliers > 0:
                questions.append(
                    f"Do the {outliers} outliers in '{col}' materially influence model parameters or statistical significance?"
                )

        # --- Missing Data Impact ---
        for col in df.columns[df.isnull().any()]:
            missing_pct = df[col].isnull().mean() * 100
            if missing_pct > 5:
                questions.append(
                    f"With {missing_pct:.1f}% missing in '{col}', is advanced imputation (e.g., MICE) preferable over deletion?"
                )

        # --- Duplicate Rows ---
        if df.duplicated().any():
            questions.append(
                "Could duplicate rows introduce bias in model coefficients or inflate statistical significance?"
            )

        return questions
