# P&L Performance Analysis and Strategic Reporting

## 1. Project Overview

This project automates the analysis of Profit and Loss (P&L) statements for a group of stores or branches. It uses Z-Score statistical analysis to identify performance outliers in various financial accounts (costs, expenses, and revenues).

The primary goal is to transform raw CSV financial data into a comprehensive, multi-page strategic PDF report. This report provides both a high-level performance ranking and a detailed qualitative analysis of the most significant deviations, helping management make informed decisions.

The script is available in two language versions:
* `main_ptbr.py`: For reports in Brazilian Portuguese.
* `main_en.py`: For reports in English.

---

## 2. Key Features

* **Automated Z-Score Calculation**: Normalizes financial data against net revenue to compare stores of different sizes fairly and calculates Z-Scores for each P&L account.
* **Outlier Identification**: Automatically flags accounts where a store's performance is statistically significant (more than 2 standard deviations from the group average).
* **Dynamic PDF Report Generation**: Creates a professional, easy-to-read PDF report that includes:
    * A cover page with the company logo and reference period.
    * A general ranking of stores based on their average Z-Score.
    * Individual performance charts for each store.
    * A detailed, educational page explaining the concept of outliers.
    * A summary table of all identified outliers.
    * A qualitative analysis of each outlier, classifying it and analyzing its historical trend.
    * Historical trend charts for critical negative outliers.
* **Historical Trend Analysis**: Loads past performance data to determine if an outlier is a one-time event, a recurring issue, or part of a worsening trend.
* **Multi-language Support**: Provides separate scripts for generating reports in Portuguese and English, including translation of financial accounts.

---

## 3. How it Works: The Methodology

The core of this analysis is the **Z-Score**. Here’s how it's applied:

1.  **Data Loading**: The script loads the P&L data for the current month from the `/input` folder and all historical P&L data from the `/input/base` folder.
2.  **Normalization**: To compare stores of different sizes, absolute values (e.g., in BRL or USD) are not used directly. Instead, each P&L account is divided by the **Net Revenue** of its respective store. This converts every value into a proportion (e.g., "Cost of Goods Sold as a percentage of Net Revenue"). This step is crucial for a fair comparison.
3.  **Z-Score Calculation**: For each normalized P&L account, the script calculates the average and standard deviation across all stores. The Z-Score for each store's account is then calculated using the formula:

    Z = \frac{(X - \mu)}{\sigma} $$

    Where:
    - $X$ is the store's normalized value for that account.
    - $\mu$ (mu) is the average of the normalized values across all stores.
    - $\sigma$ (sigma) is the standard deviation of the normalized values.

4.  **Interpretation**:
    * A **Z-Score of 0** means the store's performance is exactly average.
    * For a **cost/expense account**, a positive Z-Score is negative (higher cost than average), while a negative Z-Score is positive (lower cost than average).
    * For a **revenue/profit account**, a positive Z-Score is positive (higher revenue than average).
    * A Z-Score greater than **+2** or less than **-2** is considered a statistically significant **outlier**, indicating performance that is unusually different from the rest of the group.

---

## 4. Project Structure

```markdown
pel-analysis/
├── input/
│   ├── base/
│   │   ├── Interno P&L 01.25.csv
│   │   └── ... (arquivos de dados históricos)
│   └── Interno P&L 08.25.csv
├── output/
│   ├── Relatorio_Z_Scores_08.2025.pdf
│   └── Z_Score_Report_08.2025.pdf
├── src/
│   ├── main_en.py
│   └── main_ptbr.py
├── .env
├── icon-logotipo.png
├── README.md
└── requirements.txt


* **`/input`**: Contains the CSV file for the current month to be analyzed.
* **`/input/base`**: Contains all historical CSV files. The script automatically reads them to build trend analyses.
* **`/output`**: The default location where the generated PDF reports are saved.
* **`/src`**: Contains the main Python scripts.
* **`.env`**: A configuration file to store local paths and parameters. **This file is required.**
* **`icon-logotipo.png`**: The company logo used on the report's cover page.
* **`README.md`**: This documentation file.

---

## 5. Setup and Installation

### Prerequisites

* Python 3.8 or higher.
* Git.

### Step-by-Step Installation

1.  **Clone the repository:**
    ```bash
    git clone <your-repository-url>
    cd <your-repository-name>
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    # For Windows
    python -m venv venv
    .\venv\Scripts\activate

    # For macOS/Linux
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **Install the required libraries:**
    A `requirements.txt` file is recommended. If you don't have one, create it with the content below and run the installation command.

    **`requirements.txt`:**
    ```
    pandas
    matplotlib
    seaborn
    numpy
    python-dotenv
    scipy
    python-dateutil
    ```

    **Installation command:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Configure the environment file:**
    Create a file named `.env` in the root directory of the project. Copy the contents of the example below and replace the placeholder values with your actual file paths and settings.

    **`.env` file example:**
    ```
    # --- FILE PATHS ---
    # Path to the folder containing the CURRENT month's CSV
    caminho_arquivo="C:/path/to/your/project/input"

    # Path to the folder where the FINAL PDF report will be saved
    caminho_relatorio="C:/path/to/your/project/output"

    # Path to the folder containing all HISTORICAL CSV files
    caminho_base="C:/path/to/your/project/input/base"

    # --- REPORT PARAMETERS ---
    # List of stores/branches to include in the report, separated by commas
    filiais_to_keep="STORE1,STORE2,STORE3,STORE4"

    # Title that will appear on the report's cover page
    title="YOUR COMPANY GROUP NAME"
    ```

---

## 6. How to Run the Script

After completing the setup, running the analysis is straightforward:

1.  Ensure your current month's CSV file is inside the `/input` directory.
2.  Navigate to the `src` directory:
    ```bash
    cd src
    ```
3.  Run the desired version of the script:

    * **For the English report:**
        ```bash
        python main_en.py
        ```
    * **For the Portuguese report:**
        ```bash
        python main_ptbr.py
        ```

4.  The script will print its progress in the console and, upon completion, save the PDF report in the `/output` folder.
