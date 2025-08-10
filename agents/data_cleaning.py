import pandas as pd
import numpy as np

class DataCleaner:
    @staticmethod
    def clean_data(df):
        print("\nStarting data cleaning and preprocessing...")
        
        cleaning_report = {
            "missing_values_handled": {},
            "outliers_capped": {},
            "datatype_conversions": []
        }

        # 1. Handle Missing Values
        for col in df.columns:
            if df[col].isnull().sum() > 0:
                if pd.api.types.is_numeric_dtype(df[col]):
                    # Median if skewed, else mean
                    if abs(df[col].skew()) > 1:
                        fill_val = df[col].median()
                        method_used = "median"
                    else:
                        fill_val = df[col].mean()
                        method_used = "mean"
                elif pd.api.types.is_datetime64_any_dtype(df[col]):
                    fill_val = df[col].mode()[0]
                    method_used = "mode (date)"
                else:
                    fill_val = df[col].mode()[0]
                    method_used = "mode"
                
                df[col] = df[col].fillna(fill_val)
                cleaning_report["missing_values_handled"][col] = method_used

        # 2. Outlier Detection & Capping
        for col in df.select_dtypes(include='number'):
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outlier_count = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()

            if outlier_count > 0:
                df[col] = df[col].clip(lower_bound, upper_bound)
                cleaning_report["outliers_capped"][col] = int(outlier_count)

        # 3. Data Type Corrections
        for col in df.columns:
            if pd.api.types.is_datetime64_any_dtype(df[col]) or "date" in col.lower():
                df[col] = pd.to_datetime(df[col], errors='coerce')
                cleaning_report["datatype_conversions"].append((col, "datetime"))
            elif df[col].dtype == "object":
                df[col] = df[col].str.strip().str.lower()  # normalize strings
                df[col] = df[col].astype('category')
                cleaning_report["datatype_conversions"].append((col, "category"))

        print("Data cleaning completed.")
        return df, cleaning_report
