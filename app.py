import streamlit as st
import pandas as pd
import os
import json
import traceback
import subprocess

from agents.dataset_loader import DatasetLoader
from agents.data_cleaning import DataCleaner
from agents.data_profiler import DataProfiler
from agents.question_generator import QuestionGenerator
from agents.answer_generator import AnswerGenerator
from agents.visualize import visualization
from agents.report_generator import create_latex_report

def select_columns(df):
    cols = df.columns.tolist()
    selected_cols = st.multiselect(
        "Select columns to include in analysis (leave empty to select all):",
        options=cols,
        default=cols
    )
    return df[selected_cols] if selected_cols else df

def display_cleaning_summary(cleaning_summary):
    st.subheader("Data Cleaning Summary")
    if not isinstance(cleaning_summary, dict):
        st.write(cleaning_summary)
        return
    for section, details in cleaning_summary.items():
        st.write(f"**{section.replace('_', ' ').title()}**")
        if isinstance(details, dict):
            df_summary = pd.DataFrame(list(details.items()), columns=["Column", "Action"])
            st.table(df_summary)
        else:
            st.write(details)

def display_clean_profile(profile_json):
    # Numeric summary
    numeric_data = []
    for col, stats in profile_json.get("numeric_summary", {}).items():
        row = {"Column": col}
        row.update(stats)
        numeric_data.append(row)
    numeric_df = pd.DataFrame(numeric_data)
    
    # Categorical summary (unique counts and top 5 counts shown)
    categorical_data = []
    for col, info in profile_json.get("categorical_summary", {}).items():
        row = {
            "Column": col,
            "Unique Values": info.get("unique", None),
            "Missing Values": info.get("missing", None),
            "High Cardinality": info.get("high_cardinality", None),
        }
        counts = info.get("counts", {})
        top_counts = dict(sorted(counts.items(), key=lambda x: x[1], reverse=True)[:5])
        row["Top Categories (count)"] = ", ".join([f"{k}: {v}" for k, v in top_counts.items()])
        categorical_data.append(row)
    categorical_df = pd.DataFrame(categorical_data)
    
    # Datetime summary
    datetime_data = []
    for col, info in profile_json.get("datetime_summary", {}).items():
        row = {"Column": col}
        row.update(info)
        datetime_data.append(row)
    datetime_df = pd.DataFrame(datetime_data)
    
    # Correlations - flatten into long format
    correlations = profile_json.get("correlations", {})
    corr_rows = []
    for var1, corr_dict in correlations.items():
        for var2, corr_val in corr_dict.items():
            corr_rows.append({"Variable 1": var1, "Variable 2": var2, "Correlation": corr_val})
    corr_df = pd.DataFrame(corr_rows)
    
    return {
        "Numeric Summary": numeric_df,
        "Categorical Summary": categorical_df,
        "Datetime Summary": datetime_df,
        "Correlations": corr_df,
    }

def display_summary(summary):
    try:
        st.write("### Dataset Shape")
        st.write(f"Rows: {summary['shape'][0]}, Columns: {summary['shape'][1]}")

        st.write("### Data Types")
        # Handle either 'dtypes' or 'dtype' keys safely
        dtype_key = None
        if 'dtypes' in summary:
            dtype_key = 'dtypes'
        elif 'dtype' in summary:
            dtype_key = 'dtype'

        if dtype_key:
            dt_df = pd.DataFrame.from_dict(summary[dtype_key], orient='index', columns=['Data Type'])
            st.table(dt_df)
        else:
            st.warning("Data types information not found in profile summary.")

        st.write("### Missing Values")
        missing_df = pd.DataFrame.from_dict(summary['missing'], orient='index', columns=['Missing Count'])
        st.table(missing_df)

        if 'head' in summary:
            st.write("### Data Preview")
            st.dataframe(summary['head'])

    except Exception as e:
        st.error(f"Error displaying summary: {e}")
        st.write("Raw profile summary:")
        st.write(summary)

def display_visualizations(plot_paths):
    st.write("### Visualizations")
    cols_per_row = 2
    plot_items = list(plot_paths.items())

    for i in range(0, len(plot_items), cols_per_row):
        cols = st.columns(cols_per_row)
        for j, (name, path) in enumerate(plot_items[i:i+cols_per_row]):
            with cols[j]:
                st.image(path, caption=name)

def create_data_summary_from_profile(profile):
    data_summary = []
    numeric_summary = profile.get("numeric_summary", {})
    dtypes = profile.get("dtypes", {}) or profile.get("dtype", {}) or {}
    
    for col, stats in numeric_summary.items():
        row = {"column": col}
        row.update(stats)
        # Add dtype info if available
        if col in dtypes:
            row["dtype"] = dtypes[col]
        else:
            row["dtype"] = "Unknown"
        # Provide default for missing keys used in report
        if "non_missing" not in row:
            row["non_missing"] = stats.get("count", "N/A")
        if "missing" not in row:
            row["missing"] = stats.get("missing_count", 0)
        data_summary.append(row)
    return data_summary

def main():
    st.title("AutoStatAgent - Automated Data Analysis")

    uploaded_file = st.file_uploader("Upload your dataset (CSV, Excel, JSON):", type=['csv', 'xlsx', 'xls', 'json'])
    if uploaded_file is None:
        return

    file_ext = os.path.splitext(uploaded_file.name)[1].lower()
    file_path = "temp_uploaded_file" + file_ext
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.success("File uploaded successfully!")

    try:
        df = DatasetLoader.load_full_dataset(file_path)
        df = select_columns(df)

        st.subheader("Dataset Preview")
        st.dataframe(df.head())

        result = DataCleaner.clean_data(df)
        if isinstance(result, tuple):
            df, cleaning_summary = result
            display_cleaning_summary(cleaning_summary)
        else:
            df = result

        profile_summary_raw = DataProfiler.profile_data(df)
        profile_summary = {}

        if isinstance(profile_summary_raw, str):
            try:
                profile_summary = json.loads(profile_summary_raw)
            except Exception as e:
                st.warning(f"Failed to parse profile summary JSON: {e}")
                profile_summary = {}
        elif isinstance(profile_summary_raw, dict):
            profile_summary = profile_summary_raw

        st.subheader("Data Profile Summary")
        if profile_summary and 'numeric_summary' in profile_summary:
            summary_tables = display_clean_profile(profile_summary)

            st.write("### Numeric Summary")
            st.dataframe(summary_tables["Numeric Summary"])

            st.write("### Categorical Summary")
            st.dataframe(summary_tables["Categorical Summary"])

            st.write("### Datetime Summary")
            st.dataframe(summary_tables["Datetime Summary"])

            st.write("### Correlations")
            st.dataframe(summary_tables["Correlations"])
        else:
            st.warning("Profile summary format unexpected, displaying raw output:")
            st.write(profile_summary_raw)

        questions = QuestionGenerator.generate_questions(df)
        st.subheader("Generated Questions (first 10)")
        for q in questions[:10]:
            st.write(f"- {q}")

        answers = AnswerGenerator.answer_questions(df, questions)
        st.subheader("Sample Answers")
        for q, a in list(answers.items())[:5]:
            st.markdown(f"**Q:** {q}")
            st.markdown(f"**A:** {a}")

        plot_paths = visualization.create_visualizations(df)
        display_visualizations(plot_paths)

        hypothesis_results = {
            q: a for q, a in answers.items()
            if "Test:" in a and ("H₀" in a or "H0" in a)
        }
        data_summary = create_data_summary_from_profile(profile_summary)

    except Exception as e:
        st.error(f"Error loading or processing data: {e}")
        return

    if st.button("Generate PDF Report"):
        with st.spinner("Generating PDF report... This may take a few moments."):
            output_pdf = "AutoStatAgent_Report"
            try:
                create_latex_report(
                    summary_stats=data_summary,
                    questions_answers=answers,
                    figure_paths=list(plot_paths.values()),
                    hypothesis_results=hypothesis_results,
                    output_path=output_pdf
                )
                st.success(f"Report generated: {output_pdf}.pdf")
            except subprocess.CalledProcessError as e:
                st.error("LaTeX compilation failed!")
                st.text(e.output.decode('utf-8') if e.output else str(e))
            except Exception as e:
                st.error(f"Error generating report: {e}")

if __name__ == "__main__":
    main()
