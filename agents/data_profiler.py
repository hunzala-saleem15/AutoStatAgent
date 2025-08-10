import pandas as pd
from scipy.stats import skew, kurtosis
import numpy as np

class DataProfiler:
    @staticmethod
    def profile_data(df):
        print("\nProfiling Dataset...")

        summary = {}

        # Column type separation
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        date_cols = df.select_dtypes(include=['datetime']).columns.tolist()

        # === Numerical Summary ===
        numeric_summary = {}
        for col in numeric_cols:
            series = df[col].dropna()
            desc = series.describe(percentiles=[0.1, 0.9]).to_dict()
            desc['skewness'] = skew(series) if len(series) > 2 else None
            desc['kurtosis'] = kurtosis(series) if len(series) > 3 else None
            desc_clean = {k: (round(v, 3) if isinstance(v, (float, np.floating)) else v) 
                          for k, v in desc.items()}
            numeric_summary[col] = desc_clean
        summary['numeric_summary'] = numeric_summary

        # === Categorical Summary ===
        cat_summary = {}
        for col in cat_cols:
            counts = df[col].value_counts(dropna=False).to_dict()
            unique = df[col].nunique()
            missing = df[col].isnull().sum()
            high_cardinality = unique > (0.5 * len(df))
            cat_summary[col] = {
                'unique': int(unique),
                'counts': {str(k): int(v) for k, v in list(counts.items())[:10]},
                'missing': int(missing),
                'high_cardinality': high_cardinality
            }
        summary['categorical_summary'] = cat_summary

        # === Datetime Summary ===
        date_summary = {}
        for col in date_cols:
            series = df[col].dropna()
            if not series.empty:
                date_summary[col] = {
                    'min': series.min(),
                    'max': series.max(),
                    'range_days': (series.max() - series.min()).days
                }
        summary['datetime_summary'] = date_summary

        # === Correlations ===
        if len(numeric_cols) > 1:
            corr = df[numeric_cols].corr().round(3).to_dict()
            summary['correlations'] = corr
        else:
            summary['correlations'] = {}

        # === Missing Values ===
        missing_counts = df.isnull().sum()
        missing_percent = (missing_counts / len(df) * 100).round(2)
        summary['missing_values'] = {col: int(missing_counts[col]) for col in df.columns}
        summary['missing_percent'] = {col: float(missing_percent[col]) for col in df.columns}

        # === Missing Patterns (safe check) ===
        try:
            missing_pattern = df.isnull().astype(int).corr().round(3).to_dict()
        except Exception:
            missing_pattern = {}
        summary['missing_patterns'] = missing_pattern

        # === Duplicates ===
        summary['duplicate_rows'] = int(df.duplicated().sum())

        print("Data profiling completed.")
        return summary
