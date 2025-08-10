# AutoStatAgent

**AutoStatAgent** is an intelligent agent-based system for automated exploratory data analysis (EDA), statistical testing, visualization, and LaTeX report generation. It enables users to upload any dataset and receive a complete statistical analysis with visual insights and a professionally formatted PDF report, all without needing prior statistical or programming knowledge.

---

## Features

- Accepts datasets in CSV, Excel, and JSON formats.
- Automatically generates meaningful statistical questions.
- Performs univariate, bivariate, and multivariate analyses.
- Conducts hypothesis testing (t-test, ANOVA, chi-square).
- Creates appropriate visualizations based on data types.
- Compiles results into a LaTeX-generated PDF report.
- Modular agent-based architecture for intelligent workflow.
- Web interface built using Streamlit for easy usage.

---

## Technology Stack

| Layer            | Tools/Libraries                            |
|------------------|-------------------------------------------|
| Language         | Python 3.x                                |
| Backend          | Pandas, NumPy, SciPy, Statsmodels         |
| Visualizations   | Matplotlib, Seaborn                        |
| Agents           | Custom modular agents (DatasetLoader, DataCleaner, DataProfiler, etc.) |
| Report Engine    | LaTeX via PyLaTeX or Jinja2                |
| Web UI           | Streamlit                                  |
| File Handling    | Pandas, openpyxl                           |
| Deployment       | Streamlit Cloud (recommended)              |

---

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/hunzala-saleem15/AutoStatAgent.git
   cd AutoStatAgent
