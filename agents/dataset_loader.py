import os
import pandas as pd
import chardet
import pyreadstat

class DatasetLoader:
    SUPPORTED_FORMATS = ['.csv', '.tsv', '.xlsx', '.xls', '.json', '.parquet', '.pkl', '.sas7bdat', '.sav']

    @staticmethod
    def detect_encoding(file_path, n_lines=10000):
        with open(file_path, 'rb') as f:
            rawdata = f.read(n_lines)
        result = chardet.detect(rawdata)
        return result['encoding']

    @staticmethod
    def load_full_dataset(file_path):
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        ext = os.path.splitext(file_path)[1].lower()
        if ext not in DatasetLoader.SUPPORTED_FORMATS:
            raise ValueError(f"Unsupported file format: {ext}")

        encoding = None
        if ext in ['.csv', '.tsv']:
            encoding = DatasetLoader.detect_encoding(file_path)

        try:
            if ext == '.csv':
                df = pd.read_csv(file_path, encoding=encoding, compression='infer')
            elif ext == '.tsv':
                df = pd.read_csv(file_path, sep='\t', encoding=encoding, compression='infer')
            elif ext in ['.xlsx', '.xls']:
                df = pd.read_excel(file_path)
            elif ext == '.json':
                df = pd.read_json(file_path)
            elif ext == '.parquet':
                df = pd.read_parquet(file_path)
            elif ext == '.pkl':
                df = pd.read_pickle(file_path)
            elif ext == '.sas7bdat':
                df = pd.read_sas(file_path)
            elif ext == '.sav':
                df, _ = pyreadstat.read_sav(file_path)
        except Exception as e:
            raise RuntimeError(f"Error loading file: {e}")

        # --- Auto parse potential datetime columns ---
        for col in df.columns:
            if pd.api.types.is_object_dtype(df[col]):
                try:
                    df[col] = pd.to_datetime(df[col], errors='ignore', infer_datetime_format=True)
                except:
                    pass

        return df

    @staticmethod
    def filter_columns(df):
        print("\nAvailable columns:")
        for i, col in enumerate(df.columns, start=1):
            print(f"{i}. {col} ({df[col].dtype})")

        selected = input("\nEnter column numbers to select (comma-separated) or press Enter to select all: ").strip()

        if selected:
            try:
                indices = [int(x.strip()) - 1 for x in selected.split(',')]
                selected_cols = [df.columns[i] for i in indices]
                df = df[selected_cols]
            except (ValueError, IndexError):
                print("Invalid input! Using all columns.")
        return df

    @staticmethod
    def summarize(df):
        summary = {
            "shape": df.shape,
            "dtypes": df.dtypes.astype(str).to_dict(),
            "missing": df.isnull().sum().to_dict(),
            "head": df.head()  # Keep as DataFrame
        }
        return summary

