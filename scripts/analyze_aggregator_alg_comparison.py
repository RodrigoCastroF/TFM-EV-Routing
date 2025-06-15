"""
Comprehensive analysis script for aggregator algorithm comparison results.

This script:
1. Loads and analyzes results from aggregator algorithm comparison CSV file
2. Analyzes the performance of different ML algorithms (linear, rf, svm, cart, gbm, mlp)
3. Creates visualizations to answer key research questions:
   - Profit comparison across algorithms by number of controlled stations
   - Profit improvement over base case for each algorithm
   - Prediction accuracy (predicted vs real profit) for each algorithm
4. Compares R2 scores with competition baseline performance
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import os
import sys
from datetime import datetime
from utils.tee_output import TeeOutput


def load_and_preprocess_data(csv_file):
    """Load and preprocess the algorithm comparison CSV file."""
    if not os.path.exists(csv_file):
        raise FileNotFoundError(f"File not found: {csv_file}")
    
    print(f"Loading CSV file: {csv_file}")
    df = pd.read_csv(csv_file)
    print(f"  → Loaded {len(df)} rows")
    
    # Add number of controlled stations
    df['num_controlled'] = df['controlled_stations'].apply(
        lambda x: len(x.split('|')) if '|' in str(x) else 1
    )
    
    # Extract algorithm name from type
    df['algorithm'] = df['type'].apply(
        lambda x: x.replace('_predicted', '').replace('_real', '') if x != 'base_case' else 'base_case'
    )
    
    # Determine if it's predicted or real
    df['prediction_type'] = df['type'].apply(
        lambda x: 'predicted' if 'predicted' in x else ('real' if 'real' in x else 'base_case')
    )
    
    # Sort by number of controlled stations and combination
    df = df.sort_values(['num_controlled', 'controlled_stations', 'type']).reset_index(drop=True)
    
    return df


def get_algorithm_colors():
    """Define consistent colors for each algorithm across all plots."""
    return {
        'base_case': 'gray',
        'linear': '#1f77b4',      # Blue
        'rf': '#ff7f0e',          # Orange
        'svm': '#2ca02c',         # Green
        'cart': '#d62728',        # Red
        'gbm': '#9467bd',         # Purple
        'mlp': '#8c564b'          # Brown
    }


def create_profit_by_stations_plot(df, output_file):
    """Create profit comparison plot by number of controlled stations."""
    print("Creating Average Profit by Number of Controlled Stations plot...")
    
    # Get real profits for each algorithm and base case
    profit_data = []
    algorithms = ['linear', 'rf', 'svm', 'cart', 'gbm', 'mlp']
    
    for combination in df['controlled_stations'].unique():
        combo_data = df[df['controlled_stations'] == combination]
        num_controlled = combo_data['num_controlled'].iloc[0]
        
        # Base case
        base_case_profit = combo_data[combo_data['type'] == 'base_case']['profit'].values
        if len(base_case_profit) > 0:
            profit_data.append({
                'num_controlled': num_controlled,
                'algorithm': 'base_case',
                'profit': base_case_profit[0],
                'combination': combination
            })
        
        # Each algorithm (real profit)
        for alg in algorithms:
            alg_real_profit = combo_data[combo_data['type'] == f'{alg}_real']['profit'].values
            if len(alg_real_profit) > 0:
                profit_data.append({
                    'num_controlled': num_controlled,
                    'algorithm': alg,
                    'profit': alg_real_profit[0],
                    'combination': combination
                })
    
    profit_df = pd.DataFrame(profit_data)
    
    # Calculate average profit by number of controlled stations and algorithm
    avg_profit = profit_df.groupby(['num_controlled', 'algorithm'])['profit'].mean().reset_index()
    pivot_data = avg_profit.pivot(index='num_controlled', columns='algorithm', values='profit')
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Get colors
    colors = get_algorithm_colors()
    
    # Plot bars for each algorithm
    x = np.arange(len(pivot_data.index))
    width = 0.12
    
    algorithms_to_plot = ['base_case'] + algorithms
    readable_labels = {
        'base_case': 'Base Case',
        'linear': 'Linear Regression',
        'rf': 'Random Forest',
        'svm': 'Support Vector Machine',
        'cart': 'Decision Tree (CART)',
        'gbm': 'Gradient Boosting',
        'mlp': 'Neural Network (MLP)'
    }
    
    for i, alg in enumerate(algorithms_to_plot):
        if alg in pivot_data.columns:
            values = pivot_data[alg].values
            readable_label = readable_labels.get(alg, alg.upper())
            ax.bar(x + i * width, values, width, label=readable_label, 
                   color=colors[alg], alpha=0.8)
    
    ax.set_xlabel('Number of Controlled Stations', fontsize=12)
    ax.set_ylabel('Average Profit ($)', fontsize=12)
    ax.set_title('Average Profit by Number of Controlled Stations', fontsize=14, fontweight='bold')
    ax.set_xticks(x + width * (len(algorithms_to_plot) - 1) / 2)
    ax.set_xticklabels(pivot_data.index)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"  → Profit comparison plot saved to: {output_file}")
    plt.close()
    
    return profit_df


def create_improvement_over_baseline_plot(profit_df, output_file):
    """Create improvement over base case plot."""
    print("Creating Profit Improvement over Base Case plot...")
    
    # Calculate improvements for each combination
    improvements = []
    algorithms = ['linear', 'rf', 'svm', 'cart', 'gbm', 'mlp']
    
    for combination in profit_df['combination'].unique():
        combo_data = profit_df[profit_df['combination'] == combination]
        
        base_case_profit = combo_data[combo_data['algorithm'] == 'base_case']['profit'].values
        if len(base_case_profit) > 0:
            base_profit = base_case_profit[0]
            num_controlled = combo_data['num_controlled'].iloc[0]
            
            for alg in algorithms:
                alg_profit = combo_data[combo_data['algorithm'] == alg]['profit'].values
                if len(alg_profit) > 0:
                    improvement = alg_profit[0] - base_profit
                    improvement_pct = (alg_profit[0] - base_profit) / base_profit * 100 if base_profit != 0 else None
                    
                    improvements.append({
                        'combination': combination,
                        'algorithm': alg,
                        'improvement': improvement,
                        'improvement_pct': improvement_pct,
                        'num_controlled': num_controlled
                    })
    
    imp_df = pd.DataFrame(improvements)
    
    if imp_df.empty:
        print("  → No improvement data available")
        return imp_df
    
    # Filter out cases with None improvement_pct (base case with 0 profit)
    imp_df_filtered = imp_df[imp_df['improvement_pct'].notna()].copy()
    excluded_count = len(imp_df) - len(imp_df_filtered)
    
    if excluded_count > 0:
        print(f"  → Excluded {excluded_count} cases where base case profit was $0 from percentage calculations")
    
    # Create readable labels (define early for use in quartiles printing)
    readable_labels = {
        'linear': 'Linear Regression',
        'rf': 'Random Forest',
        'svm': 'Support Vector Machine',
        'cart': 'Decision Tree (CART)',
        'gbm': 'Gradient Boosting',
        'mlp': 'Neural Network (MLP)'
    }
    
    # Print quartiles for each algorithm (excluding None values)
    if not imp_df_filtered.empty:
        print("\n  QUARTILES FOR IMPROVEMENT OVER BASE CASE (%) - Excluding cases with $0 base profit")
        print("  " + "-" * 78)
        for alg in algorithms:
            alg_data = imp_df_filtered[imp_df_filtered['algorithm'] == alg]['improvement_pct']
            if len(alg_data) > 0:
                q1 = alg_data.quantile(0.25)
                median = alg_data.quantile(0.50)
                q3 = alg_data.quantile(0.75)
                readable_name = readable_labels.get(alg, alg.upper())
                print(f"  {readable_name}:")
                print(f"    Q1 (25th percentile): {q1:.2f}%")
                print(f"    Q2 (50th percentile/Median): {median:.2f}%")
                print(f"    Q3 (75th percentile): {q3:.2f}%")
                print(f"    Valid cases: {len(alg_data)}")
                print()
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Get colors
    colors = get_algorithm_colors()
    
    # Create palette with display names as keys
    display_colors = {readable_labels[alg]: colors[alg] for alg in algorithms if alg in readable_labels}
    
    # Rename algorithms for display
    imp_df_filtered['algorithm_display'] = imp_df_filtered['algorithm'].map(readable_labels)
    
    if not imp_df_filtered.empty:
        sns.boxplot(data=imp_df_filtered, x='algorithm_display', y='improvement_pct', 
                   hue='algorithm_display', ax=ax, palette=display_colors, legend=False)
        
        # Rotate x-axis labels for better readability
        ax.tick_params(axis='x', rotation=45)
    
    ax.set_xlabel('Algorithm', fontsize=12)
    ax.set_ylabel('Improvement over Base Case (%)', fontsize=12)
    ax.set_title('Profit Improvement over Base Case Distribution (Excluding $0 Base Cases)', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='red', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"  → Improvement over baseline plot saved to: {output_file}")
    plt.close()
    
    return imp_df


def create_prediction_accuracy_plot(df, output_file):
    """Create prediction accuracy plot."""
    print("Creating Prediction Accuracy plot...")
    
    # Get prediction vs real data
    pred_real_data = []
    algorithms = ['linear', 'rf', 'svm', 'cart', 'gbm', 'mlp']
    
    for combination in df['controlled_stations'].unique():
        combo_data = df[df['controlled_stations'] == combination]
        
        for alg in algorithms:
            predicted_profit = combo_data[combo_data['type'] == f'{alg}_predicted']['profit'].values
            real_profit = combo_data[combo_data['type'] == f'{alg}_real']['profit'].values
            
            if len(predicted_profit) > 0 and len(real_profit) > 0:
                pred_real_data.append({
                    'algorithm': alg,
                    'predicted': predicted_profit[0],
                    'real': real_profit[0],
                    'combination': combination,
                    'num_controlled': combo_data['num_controlled'].iloc[0]
                })
    
    pred_real_df = pd.DataFrame(pred_real_data)
    
    if pred_real_df.empty:
        print("  → No prediction data available")
        return pred_real_df
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Get colors
    colors = get_algorithm_colors()
    
    # Create readable labels
    readable_labels = {
        'linear': 'Linear Regression',
        'rf': 'Random Forest',
        'svm': 'Support Vector Machine',
        'cart': 'Decision Tree (CART)',
        'gbm': 'Gradient Boosting',
        'mlp': 'Neural Network (MLP)'
    }
    
    # Plot scatter for each algorithm
    for alg in algorithms:
        alg_data = pred_real_df[pred_real_df['algorithm'] == alg]
        if not alg_data.empty:
            readable_label = readable_labels.get(alg, alg.upper())
            ax.scatter(alg_data['predicted'], alg_data['real'], 
                      label=readable_label, alpha=0.7, s=60, color=colors[alg])
    
    # Add diagonal line for perfect prediction
    if not pred_real_df.empty:
        min_val = min(pred_real_df['predicted'].min(), pred_real_df['real'].min())
        max_val = max(pred_real_df['predicted'].max(), pred_real_df['real'].max())
        ax.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5, label='Perfect Prediction')
    
    ax.set_xlabel('Predicted Profit ($)', fontsize=12)
    ax.set_ylabel('Real Profit ($)', fontsize=12)
    ax.set_title('Prediction Accuracy: Predicted vs Real Profit', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"  → Prediction accuracy plot saved to: {output_file}")
    plt.close()
    
    return pred_real_df


def load_competition_performance(csv_file):
    """Load competition performance data for R2 comparison."""
    if not os.path.exists(csv_file):
        print(f"Competition performance file not found: {csv_file}")
        return None
    
    print(f"Loading competition performance data: {csv_file}")
    comp_df = pd.read_csv(csv_file)
    
    # Extract relevant columns
    comp_r2_data = []
    for _, row in comp_df.iterrows():
        # Extract station number from outcome column (e.g., 'profit_11' -> '11')
        if 'profit_' in row['outcome']:
            station = row['outcome'].replace('profit_', '')
            comp_r2_data.append({
                'station': station,
                'algorithm': row['alg'],
                'test_r2': row['test_r2']
            })
    
    return pd.DataFrame(comp_r2_data)


def r_squared(y_true, y_pred, y_mean):
    """Calculate R-squared using the same formula as in run_MLmodels.py"""
    ss_res = ((y_true - y_pred)**2).sum()
    ss_tot = ((y_true - y_mean)**2).sum()
    return (1 - (ss_res/ss_tot))


def calculate_aggregator_r2_scores(pred_real_df):
    """Calculate R2 scores for each algorithm in the aggregator context using the same formula as competition."""
    r2_scores = {}
    r2_correlation_based = {}
    r2_single_station = {}
    algorithms = pred_real_df['algorithm'].unique()
    
    for alg in algorithms:
        alg_data = pred_real_df[pred_real_df['algorithm'] == alg]
        if len(alg_data) > 1:  # Need at least 2 points for R2
            y_true = alg_data['real'].values
            y_pred = alg_data['predicted'].values
            y_mean = y_true.mean()  # Mean of the true values (real profits)
            
            # R2 using the same formula as competition
            r2 = r_squared(y_true, y_pred, y_mean)
            r2_scores[alg] = r2
            
            # R2 based on correlation (for comparison)
            correlation = np.corrcoef(y_true, y_pred)[0, 1]
            r2_correlation_based[alg] = correlation**2
        
        # R2 for single-station combinations only
        single_station_data = alg_data[alg_data['num_controlled'] == 1]
        if len(single_station_data) > 1:  # Need at least 2 points for R2
            y_true_single = single_station_data['real'].values
            y_pred_single = single_station_data['predicted'].values
            y_mean_single = y_true_single.mean()

            r2_single = r_squared(y_true_single, y_pred_single, y_mean_single)
            r2_single_station[alg] = r2_single
    
    return r2_scores, r2_correlation_based, r2_single_station


def print_summary_statistics(profit_df, imp_df, pred_real_df, aggregator_r2, aggregator_r2_corr, aggregator_r2_single, competition_r2):
    """Print comprehensive summary statistics."""
    print("\n" + "="*80)
    print("SUMMARY STATISTICS AND ANALYSIS")
    print("="*80)
    
    # 1. Profit comparison statistics
    print("\n1. PROFIT COMPARISON BY ALGORITHM")
    print("-" * 50)
    
    algorithms = ['linear', 'rf', 'svm', 'cart', 'gbm', 'mlp']
    
    if not profit_df.empty:
        for alg in ['base_case'] + algorithms:
            alg_data = profit_df[profit_df['algorithm'] == alg]['profit']
            if not alg_data.empty:
                print(f"{alg.upper()}:")
                print(f"  Average profit: ${alg_data.mean():.2f}")
                print(f"  Median profit: ${alg_data.median():.2f}")
                print(f"  Std deviation: ${alg_data.std():.2f}")
                print(f"  Min profit: ${alg_data.min():.2f}")
                print(f"  Max profit: ${alg_data.max():.2f}")
                print()
    
    # 2. Improvement over base case statistics
    print("2. IMPROVEMENT OVER BASE CASE")
    print("-" * 50)
    
    if not imp_df.empty:
        # Filter out cases with None improvement_pct
        imp_df_filtered = imp_df[imp_df['improvement_pct'].notna()].copy()
        excluded_count = len(imp_df) - len(imp_df_filtered)
        
        if excluded_count > 0:
            print(f"Note: Excluded {excluded_count} cases where base case profit was $0")
            print()
        
        for alg in algorithms:
            alg_data = imp_df[imp_df['algorithm'] == alg]
            alg_data_filtered = imp_df_filtered[imp_df_filtered['algorithm'] == alg]
            
            better_cases = (alg_data['improvement'] > 0).sum()
            worse_cases = (alg_data['improvement'] < 0).sum()
            same_cases = (alg_data['improvement'] == 0).sum()
            
            print(f"{alg.upper()}:")
            print(f"  Better than base case: {better_cases}/{len(alg_data)} cases ({better_cases/len(alg_data)*100:.1f}%)")
            print(f"  Worse than base case: {worse_cases}/{len(alg_data)} cases ({worse_cases/len(alg_data)*100:.1f}%)")
            
            if len(alg_data_filtered) > 0:
                avg_improvement = alg_data_filtered['improvement_pct'].mean()
                print(f"  Average improvement: {avg_improvement:.1f}% (from {len(alg_data_filtered)} valid cases)")
            else:
                print(f"  Average improvement: N/A (all base cases had $0 profit)")
            print()
    
    # 3. Prediction accuracy statistics
    print("3. PREDICTION ACCURACY")
    print("-" * 50)
    
    if not pred_real_df.empty:
        for alg in algorithms:
            alg_data = pred_real_df[pred_real_df['algorithm'] == alg]
            if not alg_data.empty:
                correlation = alg_data['predicted'].corr(alg_data['real'])
                mae = np.mean(np.abs(alg_data['predicted'] - alg_data['real']))
                rmse = np.sqrt(np.mean((alg_data['predicted'] - alg_data['real'])**2))
                
                print(f"{alg.upper()}:")
                print(f"  Correlation: {correlation:.3f}")
                print(f"  Mean Absolute Error: ${mae:.2f}")
                print(f"  Root Mean Square Error: ${rmse:.2f}")
                print()
    
    # 4. R2 Score comparison
    print("4. R2 SCORE COMPARISON")
    print("-" * 50)
    
    if aggregator_r2:
        print("Aggregator Context R2 Scores (using competition formula):")
        for alg, r2 in aggregator_r2.items():
            print(f"  {alg.upper()}: {r2:.3f}")
        print()
        
        print("Aggregator Context R2 Scores (correlation-based, for comparison):")
        for alg, r2_corr in aggregator_r2_corr.items():
            print(f"  {alg.upper()}: {r2_corr:.3f}")
        print()
        
        print("Aggregator Context R2 Scores for Single Stations Only (using competition formula):")
        for alg, r2_single in aggregator_r2_single.items():
            print(f"  {alg.upper()}: {r2_single:.3f}")
        print()
    
    if competition_r2 is not None:
        print("Competition Baseline R2 Scores (Average across stations):")
        comp_avg_r2 = competition_r2.groupby('algorithm')['test_r2'].mean()
        for alg, r2 in comp_avg_r2.items():
            print(f"  {alg.upper()}: {r2:.3f}")
        print()
        
        # Generate comprehensive R2 scores markdown table
        if aggregator_r2:
            print("R2 Scores Comprehensive Comparison (Markdown Table):")
            print()
            
            # Table header
            print("| Algorithm | Aggregator (Competition Formula) | Aggregator (Correlation-based) | Single Stations Only | Competition Baseline |")
            print("|-----------|----------------------------------|--------------------------------|---------------------|---------------------|")
            
            # Table rows
            for alg in algorithms:
                alg_display = alg.upper()
                
                # Aggregator R2 (competition formula)
                agg_r2 = aggregator_r2.get(alg, 'N/A')
                agg_r2_str = f"{agg_r2:.3f}" if agg_r2 != 'N/A' else 'N/A'
                
                # Aggregator R2 (correlation-based)
                agg_r2_corr = aggregator_r2_corr.get(alg, 'N/A')
                agg_r2_corr_str = f"{agg_r2_corr:.3f}" if agg_r2_corr != 'N/A' else 'N/A'
                
                # Single stations R2
                agg_r2_single = aggregator_r2_single.get(alg, 'N/A')
                agg_r2_single_str = f"{agg_r2_single:.3f}" if agg_r2_single != 'N/A' else 'N/A'
                
                # Competition baseline R2
                comp_r2_str = 'N/A'
                if competition_r2 is not None and not comp_avg_r2.empty and alg in comp_avg_r2.index:
                    comp_r2_str = f"{comp_avg_r2[alg]:.3f}"
                
                print(f"| {alg_display} | {agg_r2_str} | {agg_r2_corr_str} | {agg_r2_single_str} | {comp_r2_str} |")
            
            print()
        
        # Generate comprehensive summary statistics table
        print("5. SUMMARY STATISTICS TABLE")
        print("-" * 50)
        print()
        
        # Table header
        print("| Algorithm | Average Saved R2 | Correlation (Pred vs Real) | MAE (Pred vs Real) | Better than Base Case (%) | Avg Improvement (%) | Mean Profit ($) |")
        print("|-----------|------------------|----------------------------|--------------------|---------------------------|---------------------|-----------------|")
        
        # Calculate statistics for each algorithm
        for alg in algorithms:
            alg_display = alg.upper()
            
            # Competition baseline R2 (Average Saved R2)
            comp_r2_str = 'N/A'
            if competition_r2 is not None and not comp_avg_r2.empty and alg in comp_avg_r2.index:
                comp_r2_str = f"{comp_avg_r2[alg]:.3f}"
            
            # Correlation and MAE from prediction accuracy
            correlation_str = 'N/A'
            mae_str = 'N/A'
            if not pred_real_df.empty:
                alg_pred_data = pred_real_df[pred_real_df['algorithm'] == alg]
                if not alg_pred_data.empty:
                    correlation = alg_pred_data['predicted'].corr(alg_pred_data['real'])
                    correlation_str = f"{correlation:.3f}" if not pd.isna(correlation) else 'N/A'
                    mae = np.mean(np.abs(alg_pred_data['predicted'] - alg_pred_data['real']))
                    mae_str = f"{mae:.2f}"
            
            # Better than base case percentage and average improvement
            better_than_base_str = 'N/A'
            avg_improvement_str = 'N/A'
            if not imp_df.empty:
                alg_imp_data = imp_df[imp_df['algorithm'] == alg]
                if not alg_imp_data.empty:
                    better_cases = (alg_imp_data['improvement'] > 0).sum()
                    better_than_base_pct = better_cases / len(alg_imp_data) * 100
                    better_than_base_str = f"{better_than_base_pct:.1f}"
                    
                    # Average improvement (excluding cases with None improvement_pct)
                    alg_imp_data_filtered = alg_imp_data[alg_imp_data['improvement_pct'].notna()]
                    if not alg_imp_data_filtered.empty:
                        avg_improvement = alg_imp_data_filtered['improvement_pct'].mean()
                        avg_improvement_str = f"{avg_improvement:.1f}"
            
            # Mean profit
            mean_profit_str = 'N/A'
            if not profit_df.empty:
                alg_profit_data = profit_df[profit_df['algorithm'] == alg]
                if not alg_profit_data.empty:
                    mean_profit = alg_profit_data['profit'].mean()
                    mean_profit_str = f"{mean_profit:.2f}"
            
            print(f"| {alg_display} | {comp_r2_str} | {correlation_str} | {mae_str} | {better_than_base_str} | {avg_improvement_str} | {mean_profit_str} |")
        
        print()


def main():
    """Main function to run the complete analysis."""
    # Define input and output files
    csv_file = "../results/aggregator_37map_alg_comparison_20250615_130521.csv"
    competition_csv = "../regressors/37map_1001scenarios_competition_performance_comparison.csv"
    
    # Define output image files
    output_files = [
        "../images/aggregator_37map_alg_comparison_profit_by_stations.png",
        "../images/aggregator_37map_alg_comparison_improvement_over_baseline.png",
        "../images/aggregator_37map_alg_comparison_prediction_accuracy.png"
    ]

    # Define log file path
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file_path = f"../logs/aggregator_37map_alg_comparison_analysis_{timestamp}.txt"
    
    # Set up output redirection to both console and file
    tee_output = TeeOutput(log_file_path)
    original_stdout = sys.stdout
    sys.stdout = tee_output
    
    try:
        print("Aggregator Algorithm Comparison Analysis")
        print("=" * 50)
        
        # Load and preprocess data
        print("Loading and preprocessing data...")
        df = load_and_preprocess_data(csv_file)
        
        print(f"Data overview:")
        print(f"  Total combinations: {df['controlled_stations'].nunique()}")
        print(f"  Number of controlled stations range: {df['num_controlled'].min()}-{df['num_controlled'].max()}")
        print(f"  Algorithms: {df['algorithm'].unique().tolist()}")
        print(f"  Total records: {len(df)}")
        print()
        
        # Create visualizations
        print("Creating visualizations...")
        
        # 1. Profit by stations plot
        profit_df = create_profit_by_stations_plot(df, output_files[0])
        
        # 2. Improvement over baseline plot
        imp_df = create_improvement_over_baseline_plot(profit_df, output_files[1])
        
        # 3. Prediction accuracy plot
        pred_real_df = create_prediction_accuracy_plot(df, output_files[2])
        
        # Calculate R2 scores
        print("\nCalculating R2 scores...")
        aggregator_r2, aggregator_r2_corr, aggregator_r2_single = calculate_aggregator_r2_scores(pred_real_df)
        competition_r2 = load_competition_performance(competition_csv)
        
        # Print summary statistics
        print_summary_statistics(profit_df, imp_df, pred_real_df, aggregator_r2, aggregator_r2_corr, aggregator_r2_single, competition_r2)
        
        print("\nAnalysis completed successfully!")
        print(f"All plots saved to the images/ directory")
        
    except Exception as e:
        print(f"Error during analysis: {e}")
        import traceback
        traceback.print_exc()
        raise
    finally:
        # Restore original stdout and close the log file
        sys.stdout = original_stdout
        tee_output.close()
        print(f"All output has been saved to: {log_file_path}")


if __name__ == "__main__":
    main() 