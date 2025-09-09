# ==============================================================================
# FINAL SCRIPT FOR P&L ANALYSIS AND STRATEGIC REPORT GENERATION
# ==============================================================================

# --- Library Imports ---
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import matplotlib.lines as mlines
import matplotlib.dates as mdates
from scipy.stats import norm
import numpy as np
from dotenv import load_dotenv
import os
import sys
import glob
import re
from dateutil.relativedelta import relativedelta
import locale

# --- LOCALE CONFIGURATION (IMPORTANT FOR DATE PARSING) ---
try:
    locale.setlocale(locale.LC_TIME, 'en_US.UTF-8')
except locale.Error:
    print("WARNING: Locale en_US.UTF-8 not found. Using default system locale.")

# --- ACCOUNT TYPE DEFINITIONS ---
# List with the names of revenue/profit accounts.
REVENUE_ACCOUNTS = [
    'GROSS_SALE', 
    'NET_REVENUE',
    'GROSS_PROFIT',
    'Store_Operating_Income',
    'Interest_Income',
    'NET_INCOME_BEFORE_TAX',
    'NET_INCOME'
]
# The remaining accounts will automatically be treated as COSTS/EXPENSES, where a low Z-Score is POSITIVE.

# ==============================================================================
# --- GLOBAL UTILITY FUNCTIONS ---
# ==============================================================================

def convert_to_float(value):
    if isinstance(value, str):
        try:
            # Standard US format: 1,234.56 -> remove ','
            # Standard BR format: 1.234,56 -> remove '.', then replace ',' with '.'
            return float(value.replace('.', '').replace(',', '.'))
        except (ValueError, TypeError):
            return np.nan
    return value

# ==============================================================================
# --- PLOTTING AND TEXT GENERATION FUNCTIONS ---
# ==============================================================================

def create_cover_page(pdf_pages, title, current_date_object, emission_date, logo_path):
    fig, ax = plt.subplots(figsize=(8.5, 11))
    ax.axis('off')

    y_logo = 0.95 
    y_title_group = 0.46 
    y_subtitle_pnl = 0.41
    y_report_type = 0.35 
    y_reference_period = 0.30 
    y_emission_date = 0.1

    try:
        logo = plt.imread(logo_path)
        imagebox = OffsetImage(logo, zoom=0.6)
        ab = AnnotationBbox(imagebox, (0.5, y_logo), frameon=False, box_alignment=(0.5, 0.5))
        ax.add_artist(ab)
    except FileNotFoundError:
        print(f"WARNING: Logo file not found at '{logo_path}'")
        ax.text(0.5, y_logo, '[Logo Not Found]', ha='center', va='center', fontsize=12, color='red')

    # Cover Page Texts
    ax.text(0.5, y_title_group, title, ha='center', va='center', fontsize=28, weight='bold', color='#2F4F4F')
    ax.text(0.5, y_report_type, 'Performance and Operational Efficiency Analysis', ha='center', va='center', fontsize=18, color='#696969')

    # Reference Month Block
    ax.text(0.5, y_reference_period, 'REFERENCE MONTH', ha='center', va='center', fontsize=12, color='grey')
    ax.text(0.5, 0.27, current_date_object.strftime('%B %Y').upper(), ha='center', va='center', fontsize=24, weight='bold', color='#4CAF50')

    # Footer
    ax.text(0.5, y_emission_date, f'Emission Date: {emission_date}', ha='center', va='center', fontsize=10, color='#A9A9A9')
    ax.text(0.5, 0.07, 'CONFIDENTIAL - FOR INTERNAL USE ONLY', ha='center', va='center', fontsize=9, color='grey')

    pdf_pages.savefig(fig, bbox_inches='tight')
    plt.close(fig)

def export_dataframe_to_pdf(df, pdf_pages, title, max_rows_per_page=20):
    num_rows = len(df)
    if num_rows == 0:
        return
    num_pages = (num_rows // max_rows_per_page) + (1 if num_rows % max_rows_per_page != 0 else 0)
    for page in range(num_pages):
        start_row, end_row = page * max_rows_per_page, min((page + 1) * max_rows_per_page, num_rows)
        df_page = df.iloc[start_row:end_row]
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.axis('off')
        fig.suptitle(title, fontsize=14, weight='bold', y=0.98)
        num_cols = len(df_page.columns)
        col_widths = [0.4] + [(0.6 / (num_cols - 1))] * (num_cols - 1) if num_cols > 1 else [1.0]
        table = ax.table(cellText=df_page.values, colLabels=df_page.columns, cellLoc='center', loc='center', bbox=[0, 0, 1, 0.9], colWidths=col_widths)
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.0, 1.5)
        cells = table.get_celld()
        for i in range(len(df_page.index) + 1):
            cells[(i, 0)].set_text_props(ha='left')
            if i == 0:
                cells[(i, 0)].set_text_props(weight='bold', ha='left')
                for j in range(1, num_cols):
                    cells[(i, j)].set_text_props(weight='bold')
        pdf_pages.savefig(fig, bbox_inches='tight')
        plt.close(fig)

def plot_weighted_mean_z_scores(transposed_z_scores, stores_to_keep, pdf_pages, title, date):
    sorted_z_scores = transposed_z_scores.loc[stores_to_keep, 'ZScore_Medio_Geral'].sort_values(ascending=False)
    colors = [PALETTE["negative_light"] if x > 0 else PALETTE["positive"] for x in sorted_z_scores.values]
    fig, ax = plt.subplots(figsize=(10, 12))
    sns.barplot(y=sorted_z_scores.index, x=sorted_z_scores.values, palette=colors, ax=ax)
    ax.set_title(title, fontsize=18, weight='bold', pad=20)
    ax.set_xlabel('Average Z-Score (Relative Performance)', fontsize=14, labelpad=10)
    ax.set_ylabel('Stores', fontsize=14, labelpad=10)
    ax.axvline(0, color='grey', linewidth=1.5, linestyle='--')
    for index, value in enumerate(sorted_z_scores):
        ax.text(value, index, f' {value:.2f}', color='black', ha='left' if value >= 0 else 'right', va="center", fontsize=11)
    sns.despine(left=True, bottom=True)
    pdf_pages.savefig(fig, bbox_inches='tight')
    plt.close(fig)

def plot_individual_z_scores(transposed_z_scores, selected_stores, pdf_pages, title, date):
    for store in selected_stores:
        if store in transposed_z_scores.index:
            store_z_scores = transposed_z_scores.loc[store].drop(['ZScore_Medio_Geral'], errors='ignore')
            fig, ax = plt.subplots(figsize=(10, 14))
            colors = [PALETTE["negative_critical"] if z > 2 or z < -2 else PALETTE["negative_light"] if z > 0 else PALETTE["positive"] for z in store_z_scores.values]
            sns.barplot(y=store_z_scores.index, x=store_z_scores.values, palette=colors, ax=ax)
            ax.set_title(f'{title} - {store}', fontsize=18, weight='bold', pad=20)
            ax.set_xlabel('Z-Score')
            ax.set_ylabel('Cost and Expense Elements')
            ax.tick_params(axis='y', labelsize=12 if len(store_z_scores.index) < 30 else 10)
            ax.axvline(0, color='grey', linewidth=1.5, linestyle='--')
            for idx, value in enumerate(store_z_scores):
                ax.text(value, idx, f' {value:.2f}', color='black', ha='left' if value >= 0 else 'right', va="center")
            sns.despine(left=True, bottom=True)
            fig.tight_layout(rect=[0, 0, 1, 0.96])
            pdf_pages.savefig(fig, bbox_inches='tight')
            plt.close(fig)

def identify_outliers(z_scores, normalized_df):
    outliers_summary = []
    for store in z_scores.index:
        if store in ['mean', 'std']:
            continue
        for metric in z_scores.columns:
            if metric in ['mean', 'std', 'ZScore_Medio_Geral']:
                continue
            z_score = z_scores.loc[store, metric]
            if z_score > 2 or z_score < -2:
                outliers_summary.append({'Store': store, 'Cost/Expense Element': metric, 'Z-Score': z_score})
    return pd.DataFrame(outliers_summary)

def generate_outlier_intro_page(pdf_pages, reference_date):
    def generate_normal_distribution_chart(ax, mean, std_dev, outlier_value):
        x = np.linspace(mean - 4 * std_dev, mean + 4 * std_dev, 1000)
        pdf = norm.pdf(x, mean, std_dev)
        ax.plot(x, pdf, 'b-', lw=2, label='Normal Distribution Curve')
        upper_limit = mean + 2 * std_dev
        lower_limit = mean - 2 * std_dev
        ax.axvline(mean, color='black', linestyle='--', label='Group Average')
        ax.axvline(upper_limit, color='red', linestyle='--', label='Outlier Threshold (+/- 2 SD)')
        ax.axvline(lower_limit, color='red', linestyle='--')
        x_normal = np.linspace(lower_limit, upper_limit, 500)
        ax.fill_between(x_normal, 0, norm.pdf(x_normal, mean, std_dev), color='lightblue', alpha=0.5, label='Common Values')
        ax.axvline(outlier_value, color='orange', linestyle='-', lw=2, label='Outlier Value')
        ax.annotate(f'Outlier Value:\n{outlier_value:.2f}', xy=(outlier_value, pdf.max() * 0.1),
                    xytext=(outlier_value, pdf.max() * 0.4),
                    arrowprops=dict(facecolor='orange', shrink=0.05, width=1.5, headwidth=8),
                    ha='center', fontsize=9, bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="orange", lw=1))
        y_text_pos = pdf.max() * 0.05
        ax.text(mean, y_text_pos, f'{mean:.2f}', ha='center', va='bottom', fontsize=9, weight='bold')
        ax.text(upper_limit, y_text_pos, f'{upper_limit:.2f}', ha='center', va='bottom', fontsize=9, weight='bold', color='red')
        ax.text(lower_limit, y_text_pos, f'{lower_limit:.2f}', ha='center', va='bottom', fontsize=9, weight='bold', color='red')
        ax.set_title('Visual Example of an Outlier', fontsize=14, pad=10)
        ax.set_xlabel('Cost Indicator Value', fontsize=10)
        ax.set_ylabel('Probability Density', fontsize=10)
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.legend(fontsize=9)
        ax.tick_params(axis='both', which='major', labelsize=8)
        ax.set_yticks([])

    fig = plt.figure(figsize=(8.5, 11))
    fig.suptitle(f"Detailed Outlier Analysis - {reference_date}", fontsize=16, weight='bold', y=0.97)
    fig.text(0.5, 0.92, "How to Understand This Analysis", ha='center', fontsize=12, weight='bold')

    definition_text = (
        "1. What are Outliers?\n"
        "Outliers are values that stand out for being much larger or smaller than the average of a group. In this analysis, "
        "they help us quickly identify costs that are unusually high or low in a specific store. We consider a cost an 'outlier' "
        "if it is more than 2 standard deviations away from the group mean.\n\n"
        "The chart below illustrates this concept: values falling in the blue area are considered common, while values "
        "outside of it, like the highlighted 'Outlier Value', are investigated."
    )
    fig.text(0.08, 0.88, definition_text, ha='left', va='top', fontsize=11, wrap=True, linespacing=1.5)

    chart_ax = fig.add_axes([0.1, 0.40, 0.8, 0.3])
    generate_normal_distribution_chart(ax=chart_ax, mean=-1524.82, std_dev=591.58, outlier_value=-17.76)

    normalization_text = (
        "2. The Importance of Normalization (Comparing Stores of Different Sizes)\n"
        "To compare stores fairly, we don't look at the absolute expense (in $), as a store with high revenue will "
        "naturally have higher costs. Instead, we 'normalize' the values by converting each cost into a proportion of "
        "the store's own revenue. Thus, when we say a cost is an outlier, we are saying that the percentage of revenue "
        "that store spends on that item is very different from the average percentage spent by the rest of the group."
    )
    fig.text(0.08, 0.33, normalization_text, ha='left', va='top', fontsize=11, wrap=True, linespacing=1.5)

    pdf_pages.savefig(fig, bbox_inches='tight')
    plt.close(fig)

def generate_qualitative_outlier_analysis(pdf_pages, outliers_df, temporal_df, normalized_df, reference_date):
    if outliers_df.empty:
        print("No outliers found for qualitative analysis.")
        return

    charts_to_generate = []
    revenue_accounts_set = set(REVENUE_ACCOUNTS)
    grouped_outliers = outliers_df.groupby('Store')
    
    fig, ax = plt.subplots(figsize=(8.5, 11))
    ax.axis('off')
    y_pos = 0.92
    
    fig.text(0.5, 0.97, f"Qualitative Analysis of Outliers and Trends - {reference_date}", ha='center', fontsize=16, weight='bold')

    for store, group in grouped_outliers:
        if y_pos < 0.2:
            pdf_pages.savefig(fig, bbox_inches='tight'); plt.close(fig)
            fig, ax = plt.subplots(figsize=(8.5, 11)); ax.axis('off')
            fig.text(0.5, 0.97, f"Qualitative Outlier Analysis (continued) - {reference_date}", ha='center', fontsize=16, weight='bold')
            y_pos = 0.92

        fig.text(0.1, y_pos, f"Store: {store}", fontsize=13, weight='bold')
        y_pos -= 0.04 

        for _, outlier in group.iterrows():
            if y_pos < 0.3:
                pdf_pages.savefig(fig, bbox_inches='tight'); plt.close(fig)
                fig, ax = plt.subplots(figsize=(8.5, 11)); ax.axis('off')
                fig.text(0.5, 0.97, f"Qualitative Outlier Analysis (continued) - {reference_date}", ha='center', fontsize=16, weight='bold')
                y_pos = 0.92

            metric = outlier['Cost/Expense Element']
            z_score = outlier['Z-Score']
            
            group_mean = normalized_df.loc[metric, 'mean']
            group_std = normalized_df.loc[metric, 'std']

            is_negative = False 
            classification, classification_color, description, zscore_explanation = "", "black", "", ""
            is_revenue = metric in revenue_accounts_set

            if is_revenue:
                if z_score < -2:
                    classification = "Warning"
                    classification_color = "red"
                    is_negative = True
                    description = "Revenue/profit was significantly LOWER than the group average."
                    zscore_explanation = "This value is an outlier as it is far below the group average"
                elif z_score > 2:
                    classification = "Excellent Performance"
                    classification_color = "green"
                    description = "Revenue/profit was significantly HIGHER than the group average."
                    zscore_explanation = "This value is an outlier as it is far above the group average"
            else: # Cost/Expense
                if z_score < -3.5:
                    classification = "Excellent Performance"
                    classification_color = "green"
                    description = "Cost was EXCEPTIONALLY LOWER than the group average."
                    zscore_explanation = "This value is an outlier as it is far below the group average"
                elif z_score < -2:
                    classification = "Efficiency"
                    classification_color = "green"
                    description = "Cost was significantly LOWER than the group average."
                    zscore_explanation = "This value is an outlier as it is far below the group average"
                elif z_score > 3.5:
                    classification = "Warning"
                    classification_color = "red"
                    is_negative = True
                    description = "Cost was EXCEPTIONALLY HIGHER than the group average."
                    zscore_explanation = "This value is an outlier as it is far above the group average"
                elif z_score > 2:
                    classification = "Attention Point"
                    classification_color = "red"
                    is_negative = True
                    description = "Cost was significantly HIGHER than the group average."
                    zscore_explanation = "This value is an outlier as it is far above the group average"
            
            if is_negative:
                charts_to_generate.append({'store': store, 'metric': metric})

            hist_data = temporal_df[(temporal_df['Store'] == store) & (temporal_df['Metric'] == metric)].tail(4)
            trend_text = "Not enough historical data to analyze the trend."
            if len(hist_data) > 1:
                recent_z_scores = hist_data['ZScore'].tolist()
                last_z = recent_z_scores[-1]
                previous_mean = np.mean(recent_z_scores[:-1]) if len(recent_z_scores) > 1 else None
                
                if abs(last_z) > 2 and (previous_mean is None or abs(previous_mean) < 1.5):
                    trend_text = "This appears to be a one-time event, as performance in previous months was closer to the average."
                elif is_revenue and last_z > previous_mean:
                     trend_text = "Performance for this account has been improving compared to previous months."
                elif not is_revenue and last_z < previous_mean:
                    trend_text = "The cost for this account shows an improving (decreasing) trend compared to previous months."
                else:
                    trend_text = "Performance for this account has been consistently different from the group average in recent months."

            actual_value = normalized_df.loc[metric, store]
            full_zscore_explanation = f"{zscore_explanation} (Group Avg: {group_mean:.2f}, SD: {group_std:.2f})."

            fig.text(0.1, y_pos, f"• Account: ", ha='left', va='top', fontsize=10); fig.text(0.18, y_pos, f"'{metric}'", ha='left', va='top', fontsize=10, weight='bold'); y_pos -= 0.025
            z_score_text = f"  Z-Score: {z_score:.2f} | "; fig.text(0.1, y_pos, z_score_text, ha='left', va='top', fontsize=10); fig.text(0.1 + (len(z_score_text) * 0.009), y_pos, classification, ha='left', va='top', fontsize=10, weight='bold', color=classification_color); y_pos -= 0.025
            fig.text(0.1, y_pos, f"  Analysis: ", ha='left', va='top', fontsize=10, weight='bold'); fig.text(0.2, y_pos, description, ha='left', va='top', fontsize=10, style='italic'); y_pos -= 0.025
            fig.text(0.1, y_pos, f"  Z-Score Detail: ", ha='left', va='top', fontsize=10, weight='bold'); fig.text(0.26, y_pos, full_zscore_explanation, ha='left', va='top', fontsize=10, style='italic', wrap=True); y_pos -= 0.035
            fig.text(0.1, y_pos, f"  Values: ", ha='left', va='top', fontsize=10, weight='bold'); fig.text(0.2, y_pos, f"(Your normalized value: {actual_value:.2f} vs. Group Avg: {group_mean:.2f})", ha='left', va='top', fontsize=10, style='italic'); y_pos -= 0.025
            fig.text(0.1, y_pos, f"  Trend: ", ha='left', va='top', fontsize=10, weight='bold'); fig.text(0.23, y_pos, trend_text, ha='left', va='top', fontsize=10, style='italic'); y_pos -= 0.05
   
    pdf_pages.savefig(fig, bbox_inches='tight')
    plt.close(fig)

    for item in charts_to_generate:
        plot_negative_outlier_trend(pdf_pages, item['store'], item['metric'], temporal_df)

def plot_negative_outlier_trend(pdf_pages, store, metric, temporal_df):
    print(f"Generating trend chart for outlier: {store} - {metric}")
    
    chart_data = temporal_df[(temporal_df['Store'] == store) & (temporal_df['Metric'] == metric)].tail(12)
    
    if len(chart_data) < 2:
        print("  -> Insufficient historical data to generate chart.")
        return

    fig, ax = plt.subplots(figsize=(12, 7))
    sns.lineplot(data=chart_data, x='Date', y='ZScore', marker='o', ax=ax, lw=2.5, markersize=8)

    ax.axhline(0, color='grey', linestyle='--', label='Group Average')
    ax.axhline(2, color='red', linestyle=':', lw=1.5, label='Negative Outlier Threshold (+2 SD)')
    ax.axhline(-2, color='green', linestyle=':', lw=1.5, label='Positive Outlier Threshold (-2 SD)')
    
    ax.fill_between(chart_data['Date'], 2, ax.get_ylim()[1], color='red', alpha=0.1, label='Warning Zone')
    
    ax.set_title(f"Z-Score Trend: {metric}\nStore: {store}", fontsize=16, weight='bold', pad=20)
    ax.set_xlabel('Reference Month', fontsize=12)
    ax.set_ylabel('Z-Score (Relative Performance)', fontsize=12)
    ax.legend()
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)

    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%y'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    fig.autofmt_xdate()

    plt.tight_layout()
    pdf_pages.savefig(fig)
    plt.close(fig)

# ==============================================================================
# --- MAIN HISTORICAL DATA LOADING FUNCTION ---
# ==============================================================================

def load_and_process_historical_data(base_path):
    print(f"Reading historical files from: {base_path}")
    csv_files = glob.glob(os.path.join(base_path, '*.csv'))
    complete_historical_data = {}
    
    for file in csv_files:
        base_name = os.path.basename(file)
        try:
            match = re.search(r'(\d{2}\.\d{2})', base_name)
            if not match:
                raise ValueError("Date pattern not found")
            date_str = match.group(1)
            month_date = datetime.strptime(date_str, '%m.%y')
        except Exception as e:
            print(f"WARNING: Could not extract date from file '{base_name}'. Skipping. (Error: {e})")
            continue
        print(f"Processing {base_name}...")
        month_df = pd.read_csv(file, delimiter=";")
        month_df = month_df.map(lambda x: x.strip() if isinstance(x, str) else x)
        month_df = month_df.drop('Grupo_Filial', axis='columns', errors='ignore')

        try: 
            columns_to_keep = month_df.columns[:month_df.columns.get_loc('Store_Operating_Income') + 1]
        except KeyError:
            print(f"WARNING: Column 'Store_Operating_Income' not found in '{base_name}'. Skipping.")
            continue
        filtered_df = month_df[columns_to_keep].copy()
        filtered_df = filtered_df.drop_duplicates(subset=[filtered_df.columns[0]], keep='first')
        for col in filtered_df.columns[1:]:
            filtered_df[col] = filtered_df[col].apply(convert_to_float)
        transposed_df = filtered_df.set_index('Filial').transpose()

        if 'NET_REVENUE' not in transposed_df.index:
            print(f"WARNING: 'NET_REVENUE' not found in '{base_name}'. Skipping.")
            continue
        net_revenue = transposed_df.loc['NET_REVENUE'].replace({0: np.nan})
        if net_revenue.isnull().all():
            print(f"WARNING: 'NET_REVENUE' has null values in '{base_name}'. Skipping.")
            continue
        normalized_df = transposed_df.div(net_revenue, axis=1) * 1000
        normalized_df['mean'] = normalized_df.mean(axis=1, skipna=True)
        normalized_df['std'] = normalized_df.std(axis=1, skipna=True)
        month_z_scores = normalized_df.apply(lambda row: (row - row['mean']) / row['std'] if row['std'] > 1e-6 else 0, axis=1)
        month_z_scores.loc['ZScore_Medio_Geral'] = month_z_scores.drop(columns=['mean', 'std']).mean()
        
        complete_historical_data[month_date] = {"z_scores": month_z_scores, "normalized": normalized_df, "raw": transposed_df}
    return complete_historical_data

# ==============================================================================
# --- MAIN EXECUTION BLOCK ---
# ==============================================================================

load_dotenv()
file_path = os.getenv("caminho_arquivo")
report_path = os.getenv("caminho_relatorio")
base_path = os.getenv("caminho_base")
logo_path = 'icon-logotipo.png'
stores_to_keep = os.getenv('filiais_to_keep').split(',')
title = os.getenv("title")
emission_date = datetime.now().strftime('%Y-%m-%d')
PALETTE = {"neutral": "#005f73", "positive": "#ee9b00" , "negative_light": "#0a9396", "negative_critical": "#ae2012"}
sns.set_theme(style="whitegrid")
sns.set_context("notebook", font_scale=1.2)

historical_data = load_and_process_historical_data(base_path)
if not historical_data:
    historical_data = {}

print("\nProcessing current month's file...")
current_filename = next((f for f in os.listdir(file_path) if f.lower().endswith('.csv')), None)
if not current_filename:
    print(f"ERROR: No .csv file found in '{file_path}'")
    sys.exit()
full_path = os.path.join(file_path, current_filename)
current_df = pd.read_csv(full_path, delimiter=";")

reference_str = current_df['End Date'].iloc[0]
current_date_object = datetime.strptime(reference_str, '%b/%y')
reference_for_filename = current_date_object.strftime('%m.%Y')
dynamic_filename = f"Z_Score_Report_{reference_for_filename}.pdf"
pdf_path_adjusted = os.path.join(report_path, dynamic_filename)

current_df = current_df.map(lambda x: x.strip() if isinstance(x, str) else x)
current_df = current_df.drop('Grupo_Filial', axis='columns', errors='ignore')
columns_to_keep = current_df.columns[:current_df.columns.get_loc('Store_Operating_Income') + 1]
current_filtered_df = current_df[columns_to_keep].copy()
current_filtered_df = current_filtered_df.drop_duplicates(subset=[current_filtered_df.columns[0]], keep='first')
for col in current_filtered_df.columns[1:]:
    current_filtered_df[col] = current_filtered_df[col].apply(convert_to_float)

current_transposed_df = current_filtered_df.set_index('Filial').transpose()
current_net_revenue = current_transposed_df.loc['NET_REVENUE'].replace({0: np.nan})
current_normalized_df = current_transposed_df.div(current_net_revenue, axis=1) * 1000
current_normalized_df['mean'] = current_normalized_df.mean(axis=1, skipna=True)
current_normalized_df['std'] = current_normalized_df.std(axis=1, skipna=True)
current_z_scores = current_normalized_df.apply(lambda row: (row - row['mean']) / row['std'] if row['std'] > 1e-6 else 0, axis=1)
current_z_scores.loc['ZScore_Medio_Geral'] = current_z_scores.drop(columns=['mean', 'std']).mean()
historical_data[current_date_object] = {"z_scores": current_z_scores, "normalized": current_normalized_df, "raw": current_transposed_df}

print("\n[INFO] Historical base and current month data have been unified.")

current_z_scores_transposed = current_z_scores.drop(columns=['mean', 'std']).transpose()

temporal_dfs_list = []
for event_date, month_data in historical_data.items():
    long_df = month_data['z_scores'].stack().reset_index()
    long_df.columns = ['Metric', 'Store', 'ZScore']
    long_df['Date'] = event_date
    temporal_dfs_list.append(long_df)
complete_temporal_df = pd.concat(temporal_dfs_list, ignore_index=True)

if current_date_object.month == 1:
    filter_start_date = current_date_object - relativedelta(months=3)
    filtered_temporal_df = complete_temporal_df[complete_temporal_df['Date'] >= filter_start_date]
else:
    filtered_temporal_df = complete_temporal_df[complete_temporal_df['Date'].dt.year == current_date_object.year]

print(f"Generating strategic report at: {pdf_path_adjusted}")

with PdfPages(pdf_path_adjusted) as pdf_pages:
    create_cover_page(pdf_pages, title, current_date_object, emission_date, logo_path)

    print("Generating snapshot analyses for the current month...")
    plot_weighted_mean_z_scores(current_z_scores_transposed, stores_to_keep, pdf_pages, f"Overall Ranking ({reference_for_filename})", emission_date)
    plot_individual_z_scores(current_z_scores_transposed, stores_to_keep, pdf_pages, f"Individual Analysis ({reference_for_filename})", emission_date)
    
    outliers_df = identify_outliers(current_z_scores_transposed, current_normalized_df)
    outliers_df = outliers_df.round(2)
    
    generate_outlier_intro_page(pdf_pages, reference_for_filename)
    
    outliers_summary = outliers_df[['Store', 'Cost/Expense Element', 'Z-Score']].sort_values(by='Z-Score', key=abs, ascending=False)
    export_dataframe_to_pdf(outliers_summary, pdf_pages, f"Numerical Outlier Summary for the Month ({reference_for_filename})")

    print("Generating qualitative outlier and trend analysis...")
    generate_qualitative_outlier_analysis(pdf_pages, outliers_df, complete_temporal_df, current_normalized_df, reference_for_filename)
    
print("Strategic report generated successfully!")