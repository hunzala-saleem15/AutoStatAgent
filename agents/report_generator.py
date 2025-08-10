from pylatex import Document, Section, Subsection, Figure, NoEscape, NewPage, Command
from pylatex.utils import bold
from pylatex import MiniPage, Figure, NoEscape, NewPage
import os
import streamlit as st

def create_latex_report(
    summary_stats,
    questions_answers,
    figure_paths,
    hypothesis_results,
    output_path="AutoStatAgent_Report"
):
    # Initialize Document with geometry for nicer margins
    geometry_options = {"margin": "1in"}
    doc = Document(geometry_options=geometry_options)

    # Title Page
    doc.preamble.append(NoEscape(r'\title{AutoStatAgent: An Agent-Based System for Automated Data Analysis and LaTeX Report Generation}'))
    doc.preamble.append(NoEscape(r'\author{AutoStatAgent System}'))
    doc.preamble.append(NoEscape(r'\date{\today}'))
    doc.append(NoEscape(r'\maketitle'))
    
    # Abstract
    with doc.create(Section("Abstract", numbering=False)):
        abstract_text = (
            "The project proposes an intelligent agentic system, AutoStatAgent, that automatically performs "
            "complete exploratory data analysis (EDA) and statistical testing on any given dataset. By "
            "leveraging autonomous agents, the system generates and answers relevant analytics questions, "
            "visualizes insights using smart aesthetics, and compiles the results into a professional LaTeX-based PDF report. "
            "The system democratizes data analytics by allowing users to simply upload a dataset without needing any statistical or programming background."
        )
        doc.append(abstract_text)
    
    doc.append(NewPage())
    
    # Introduction Section
    with doc.create(Section("Introduction")):
        intro_text = (
            "Many individuals, especially in academia and business, possess valuable datasets but lack the "
            "statistical knowledge or programming skills to derive insights. There is a need for a system that "
            "not only analyzes data but also presents results in a readable, interpretable, and publishable format—fully automated."
        )
        doc.append(intro_text)

    # Dataset Overview
    with doc.create(Section("Dataset Overview")):
        for col_summary in summary_stats:
            doc.append(bold(f"Column: {col_summary['column']}"))
            doc.append(f" | Type: {col_summary['dtype']} | Non-missing: {col_summary['non_missing']} | Missing: {col_summary['missing']}\n\n")

    # Questions & Answers Section
    with doc.create(Section("Questions & Answers")):
        for question, answer in questions_answers.items():
            with doc.create(Subsection(question)):
                doc.append(answer)

    doc.preamble.append(NoEscape(r'\usepackage{float}'))

    # Visualizations Section
    doc.preamble.append(NoEscape(r'\usepackage{float}'))

    with doc.create(Section("Visualizations")):
        for i in range(0, len(figure_paths), 2):
            with doc.create(MiniPage(width=NoEscape(r'0.48\textwidth'), pos='c')) as left_page:
                fig_path = figure_paths[i]
                if os.path.exists(fig_path):
                    with left_page.create(Figure(position='H')) as plot:
                        plot.add_image(fig_path, width=NoEscape(r'\linewidth'))
                        plot.add_caption(f"Visualization: {os.path.basename(fig_path)}")
                else:
                    left_page.append(f"Figure not found: {fig_path}")

            if i + 1 < len(figure_paths):
                with doc.create(MiniPage(width=NoEscape(r'0.48\textwidth'), pos='c')) as right_page:
                    fig_path = figure_paths[i + 1]
                    if os.path.exists(fig_path):
                        with right_page.create(Figure(position='H')) as plot:
                            plot.add_image(fig_path, width=NoEscape(r'\linewidth'))
                            plot.add_caption(f"Visualization: {os.path.basename(fig_path)}")
                    else:
                        right_page.append(f"Figure not found: {fig_path}")

            doc.append(NoEscape(r'\vspace{10pt}'))
            doc.append(NoEscape(r'\\'))  # new line for next row

    # Hypothesis Testing Results
    with doc.create(Section("Hypothesis Testing Results")):
        if hypothesis_results:
            for test_name, result in hypothesis_results.items():
                with doc.create(Subsection(test_name)):
                    doc.append(result)
        else:
            doc.append("No hypothesis testing results were generated.")
    
    # Generate PDF and keep .tex file
    import subprocess

    try:
        doc.generate_pdf(output_path, clean_tex=False, compiler='pdflatex')
    except subprocess.CalledProcessError as e:
        # Capture and show the LaTeX error output
        st.error("LaTeX compilation failed!")
        st.text(e.output.decode('utf-8') if e.output else str(e))
        raise

    print(f"Report generated: {output_path}.pdf")

