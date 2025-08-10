from agents.dataset_loader import DatasetLoader
from agents.data_cleaning import DataCleaner
from agents.data_profiler import DataProfiler
from agents.question_generator import QuestionGenerator
from agents.answer_generator import AnswerGenerator
from agents.visualize import visualization
from agents.report_generator import create_latex_report
import os
from jinja2 import Environment, FileSystemLoader
import subprocess

def main():
    try:
        print("Welcome to the AutoStatAgent!")
        print("This tool will help you analyze your dataset step by step.\n")

        dataset_path = input("Enter dataset file path: ").strip()

        # === Step 1: Load Dataset ===
        df = DatasetLoader.load_full_dataset(dataset_path)
        df = DatasetLoader.filter_columns(df)
        initial_summary = DatasetLoader.summarize(df)

        print("\nDataset Loaded Successfully!")
        print(f"Shape: {initial_summary['shape']}")
        print(f"Column Types:\n{initial_summary['dtypes']}")
        print(f"Missing Values:\n{initial_summary['missing']}\n")
        print("Preview of dataset:")
        print(initial_summary['head'].to_string(index=False))

        data_summary = []
        for col in df.columns:
            dtype = df[col].dtype
            nonmiss = df[col].notna().sum()
            miss = df[col].isna().sum()
            data_summary.append({
                'column': col,
                'dtype': str(dtype),
                'non_missing': nonmiss,
                'missing': miss
            })

        # === Step 2: Clean Data ===
        result = DataCleaner.clean_data(df)
        if isinstance(result, tuple):
            df, cleaning_summary = result
        else:
            df = result

        # === Step 3: Profile Data ===
        profile_summary = DataProfiler.profile_data(df)

        # === Step 4: Generate Questions ===
        questions = QuestionGenerator.generate_questions(df)
        print("\nGenerated Questions (first 10 shown):")
        for q in questions[:10]:
            print(f"- {q}")

        # === Step 5: Answer Questions ===
        answers = AnswerGenerator.answer_questions(df, questions)

        # Extract hypothesis results based on new format
        hypothesis_results = [
            (q, a) for q, a in answers.items()
            if "Test:" in a and ("H₀" in a or "H0" in a)
        ]

        print("\nSample Generated Answers:")
        for q, a in list(answers.items())[:5]:
            print(f"Q: {q}")
            print(f"A: {a}\n")

        # === Step 6: Visualizations ===
        plot_paths = visualization.create_visualizations(df)
        print("\nVisualizations created:")
        for name, path in plot_paths.items():
            print(f"{name} -> {path}")

        # === Step 7: Create LaTeX Report ===
        create_latex_report(
            summary_stats=data_summary,           # Your summary list of dicts per column
            questions_answers=answers,            # Dict {question: answer}
            figure_paths=list(plot_paths.values()),  # List of visualization file paths
            hypothesis_results=dict(hypothesis_results)  # Dict of hypothesis test Q&A results
        )

    except Exception as e:
        print(f"Error occurred: {e}")

if __name__ == "__main__":
    main()