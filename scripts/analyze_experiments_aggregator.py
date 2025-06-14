"""
Comprehensive analysis script for aggregator experiments results.

This script:
1. Loads and combines results from multiple experiment CSV files
2. Analyzes the performance of different aggregator model configurations
3. Creates visualizations to answer key research questions:
   - Prediction accuracy (predicted vs real profit)
   - Trust region effectiveness
   - Comparison with baseline scenarios
   - Impact of number of controlled stations
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import os
from datetime import datetime


def load_and_combine_results(csv_files):
    """Load and combine specific aggregator experiment CSV files."""
    if not csv_files:
        raise ValueError("No CSV files provided")
    
    print(f"Loading {len(csv_files)} CSV files:")
    for file in csv_files:
        print(f"  - {file}")
    
    # Load and combine all files
    all_data = []
    for file_path in csv_files:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        df = pd.read_csv(file_path)
        df['source_file'] = os.path.basename(file_path)
        all_data.append(df)
        print(f"    → Loaded {len(df)} rows from {os.path.basename(file_path)}")
    
    combined_df = pd.concat(all_data, ignore_index=True)
    print(f"\nCombined dataset: {len(combined_df)} rows")
    
    return combined_df


def preprocess_data(df):
    """Preprocess the data for analysis."""
    # Add number of controlled stations
    df['num_controlled'] = df['controlled_stations'].apply(
        lambda x: len(x.split('|')) if isinstance(x, str) else 0
    )
    
    # Parse controlled stations list
    df['controlled_stations_list'] = df['controlled_stations'].apply(
        lambda x: [int(s) for s in x.split('|')] if isinstance(x, str) else []
    )
    
    # Create a readable combination label
    df['combination_label'] = df['controlled_stations'].apply(
        lambda x: f"[{x.replace('|', ',')}]" if isinstance(x, str) else "[]"
    )
    
    # Sort by number of controlled stations and combination
    df = df.sort_values(['num_controlled', 'controlled_stations', 'type']).reset_index(drop=True)
    
    return df


def create_comprehensive_analysis(df, output_dir="../images"):
    """Create comprehensive visualizations for the analysis."""
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Set up the plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Create the main comprehensive plot
    fig, axes = plt.subplots(2, 2, figsize=(20, 16))
    fig.suptitle('Aggregator Model Performance Analysis', fontsize=16, fontweight='bold')
    
    # 1. Profit comparison by number of controlled stations
    ax1 = axes[0, 0]
    profit_by_size = df.groupby(['num_controlled', 'type'])['profit'].mean().reset_index()
    
    # Create pivot table for easier plotting
    pivot_data = profit_by_size.pivot(index='num_controlled', columns='type', values='profit')
    
    # Define main bar types (excluding predicted values)
    main_bar_types = ['base_case', 'max_prices', 'sol_real', 'sol_tr_real']
    colors = ['gray', 'lightcoral', 'blue', 'green']
    
    # Filter to only include available types
    available_types = [pt for pt in main_bar_types if pt in pivot_data.columns]
    type_colors = {pt: colors[i] for i, pt in enumerate(main_bar_types) if pt in available_types}
    
    # Plot main bars
    x = np.arange(len(pivot_data.index))
    width = 0.18
    
    bar_positions = {}
    # Create readable labels for profit types
    readable_labels = {
        'base_case': 'Base Case',
        'max_prices': 'Max Prices', 
        'sol_real': 'Solution (No Trust Region) Real Profit',
        'sol_tr_real': 'Solution (Trust Region) Real Profit'
    }
    
    for i, profit_type in enumerate(available_types):
        if profit_type in pivot_data.columns:
            values = pivot_data[profit_type].values
            readable_label = readable_labels.get(profit_type, profit_type.replace('_', ' ').title())
            bars = ax1.bar(x + i * width, values, width, label=readable_label, 
                          color=type_colors[profit_type], alpha=0.8)
            bar_positions[profit_type] = x + i * width
    
    # Add predicted values as horizontal lines over the corresponding real bars
    line_width = width * 0.8
    
    # Add predicted line for sol_real bars
    if 'sol_real' in pivot_data.columns and 'sol_predicted' in pivot_data.columns:
        sol_real_pos = bar_positions['sol_real']
        predicted_values = pivot_data['sol_predicted'].values
        for i, (pos, pred_val) in enumerate(zip(sol_real_pos, predicted_values)):
            if not np.isnan(pred_val):
                ax1.plot([pos - line_width/2, pos + line_width/2], [pred_val, pred_val], 
                        color='lightblue', linewidth=3, alpha=0.9)
        # Add legend entry
        ax1.plot([], [], color='lightblue', linewidth=3, label='Solution (No Trust Region) Predicted Profit', alpha=0.9)
    
    # Add predicted line for sol_tr_real bars
    if 'sol_tr_real' in pivot_data.columns and 'sol_tr_predicted' in pivot_data.columns:
        sol_tr_real_pos = bar_positions['sol_tr_real']
        tr_predicted_values = pivot_data['sol_tr_predicted'].values
        for i, (pos, pred_val) in enumerate(zip(sol_tr_real_pos, tr_predicted_values)):
            if not np.isnan(pred_val):
                ax1.plot([pos - line_width/2, pos + line_width/2], [pred_val, pred_val], 
                        color='lightgreen', linewidth=3, alpha=0.9)
        # Add legend entry
        ax1.plot([], [], color='lightgreen', linewidth=3, label='Solution (Trust Region) Predicted Profit', alpha=0.9)
    
    ax1.set_xlabel('Number of Controlled Stations')
    ax1.set_ylabel('Average Profit ($)')
    ax1.set_title('Average Profit by Number of Controlled Stations')
    ax1.set_xticks(x + width * (len(available_types) - 1) / 2)
    ax1.set_xticklabels(pivot_data.index)
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # 2. Prediction accuracy scatter plot
    ax2 = axes[0, 1]
    
    # Get prediction vs real data
    pred_real_data = []
    for combination in df['controlled_stations'].unique():
        combo_data = df[df['controlled_stations'] == combination]
        
        # Trust region comparison
        tr_pred = combo_data[combo_data['type'] == 'sol_tr_predicted']['profit'].values
        tr_real = combo_data[combo_data['type'] == 'sol_tr_real']['profit'].values
        if len(tr_pred) > 0 and len(tr_real) > 0:
            pred_real_data.append({
                'predicted': tr_pred[0], 'real': tr_real[0], 
                'model': 'With Trust Region', 'combination': combination,
                'num_controlled': combo_data['num_controlled'].iloc[0]
            })
        
        # No trust region comparison
        no_tr_pred = combo_data[combo_data['type'] == 'sol_predicted']['profit'].values
        no_tr_real = combo_data[combo_data['type'] == 'sol_real']['profit'].values
        if len(no_tr_pred) > 0 and len(no_tr_real) > 0:
            pred_real_data.append({
                'predicted': no_tr_pred[0], 'real': no_tr_real[0], 
                'model': 'No Trust Region', 'combination': combination,
                'num_controlled': combo_data['num_controlled'].iloc[0]
            })
    
    pred_real_df = pd.DataFrame(pred_real_data)
    
    if not pred_real_df.empty:
        # Create scatter plot with consistent colors
        model_colors = {
            'With Trust Region': 'green',
            'No Trust Region': 'blue'
        }
        for model_type in pred_real_df['model'].unique():
            model_data = pred_real_df[pred_real_df['model'] == model_type]
            color = model_colors.get(model_type, 'gray')
            ax2.scatter(model_data['predicted'], model_data['real'], 
                       label=model_type, alpha=0.7, s=60, color=color)
        
        # Add diagonal line for perfect prediction
        min_val = min(pred_real_df['predicted'].min(), pred_real_df['real'].min())
        max_val = max(pred_real_df['predicted'].max(), pred_real_df['real'].max())
        ax2.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5, label='Perfect Prediction')
        
        ax2.set_xlabel('Predicted Profit ($)')
        ax2.set_ylabel('Real Profit ($)')
        ax2.set_title('Prediction Accuracy: Predicted vs Real Profit')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    
    # 3. Trust region effectiveness
    ax3 = axes[1, 0]
    
    # Compare trust region vs no trust region real profits
    tr_comparison = []
    for combination in df['controlled_stations'].unique():
        combo_data = df[df['controlled_stations'] == combination]
        
        tr_real = combo_data[combo_data['type'] == 'sol_tr_real']['profit'].values
        no_tr_real = combo_data[combo_data['type'] == 'sol_real']['profit'].values
        
        if len(tr_real) > 0 and len(no_tr_real) > 0:
            tr_comparison.append({
                'combination': combination,
                'with_tr': tr_real[0],
                'without_tr': no_tr_real[0],
                'improvement': tr_real[0] - no_tr_real[0],
                'num_controlled': combo_data['num_controlled'].iloc[0]
            })
    
    tr_comp_df = pd.DataFrame(tr_comparison)
    
    if not tr_comp_df.empty:
        # Group by number of controlled stations
        tr_by_size = tr_comp_df.groupby('num_controlled').agg({
            'with_tr': 'mean',
            'without_tr': 'mean',
            'improvement': 'mean'
        }).reset_index()
        
        x = np.arange(len(tr_by_size))
        width = 0.35
        
        ax3.bar(x - width/2, tr_by_size['without_tr'], width, label='Without Trust Region', 
               color='blue', alpha=0.7)
        ax3.bar(x + width/2, tr_by_size['with_tr'], width, label='With Trust Region', 
               color='green', alpha=0.7)
        
        ax3.set_xlabel('Number of Controlled Stations')
        ax3.set_ylabel('Average Real Profit ($)')
        ax3.set_title('Trust Region Effectiveness')
        ax3.set_xticks(x)
        ax3.set_xticklabels(tr_by_size['num_controlled'])
        ax3.legend()
        ax3.grid(True, alpha=0.3)
    
    # 4. Improvement over baselines
    ax4 = axes[1, 1]
    
    # Calculate improvements over base case
    improvements = []
    for combination in df['controlled_stations'].unique():
        combo_data = df[df['controlled_stations'] == combination]
        
        base_case = combo_data[combo_data['type'] == 'base_case']['profit'].values
        tr_real = combo_data[combo_data['type'] == 'sol_tr_real']['profit'].values
        no_tr_real = combo_data[combo_data['type'] == 'sol_real']['profit'].values
        max_prices = combo_data[combo_data['type'] == 'max_prices']['profit'].values
        
        if len(base_case) > 0:
            base_profit = base_case[0]
            num_controlled = combo_data['num_controlled'].iloc[0]
            
            if len(tr_real) > 0:
                improvements.append({
                    'combination': combination,
                    'method': 'Aggregator (Trust Region)',
                    'improvement': tr_real[0] - base_profit,
                    'improvement_pct': (tr_real[0] - base_profit) / base_profit * 100 if base_profit != 0 else 0,
                    'num_controlled': num_controlled
                })
            
            if len(no_tr_real) > 0:
                improvements.append({
                    'combination': combination,
                    'method': 'Aggregator (No Trust Region)',
                    'improvement': no_tr_real[0] - base_profit,
                    'improvement_pct': (no_tr_real[0] - base_profit) / base_profit * 100 if base_profit != 0 else 0,
                    'num_controlled': num_controlled
                })
            
            if len(max_prices) > 0:
                improvements.append({
                    'combination': combination,
                    'method': 'Max Prices',
                    'improvement': max_prices[0] - base_profit,
                    'improvement_pct': (max_prices[0] - base_profit) / base_profit * 100 if base_profit != 0 else 0,
                    'num_controlled': num_controlled
                })
    
    imp_df = pd.DataFrame(improvements)
    
    if not imp_df.empty:
        # Box plot showing improvement distribution by method with consistent colors
        method_colors = {
            'Aggregator (Trust Region)': 'green',
            'Aggregator (No Trust Region)': 'blue', 
            'Max Prices': 'red'
        }
        
        sns.boxplot(data=imp_df, x='method', y='improvement_pct', hue='method', ax=ax4, 
                   palette=method_colors, legend=False)
        ax4.set_xlabel('Method')
        ax4.set_ylabel('Improvement over Base Case (%)')
        ax4.set_title('Profit Improvement over Base Case')
        ax4.grid(True, alpha=0.3)
        ax4.axhline(y=0, color='red', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    
    # Save the comprehensive plot
    output_file = os.path.join(output_dir, "aggregator_analysis_comprehensive.png")
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Comprehensive analysis saved to: {output_file}")
    
    plt.show()
    
    # Save individual plots
    print("\nSaving individual plots...")
    
    # 1. Save profit comparison plot
    fig1, ax = plt.subplots(figsize=(12, 8))
    profit_by_size = df.groupby(['num_controlled', 'type'])['profit'].mean().reset_index()
    pivot_data = profit_by_size.pivot(index='num_controlled', columns='type', values='profit')
    
    # Define main bar types (excluding predicted values)
    main_bar_types = ['base_case', 'max_prices', 'sol_real', 'sol_tr_real']
    colors = ['gray', 'lightcoral', 'blue', 'green']
    available_types = [pt for pt in main_bar_types if pt in pivot_data.columns]
    type_colors = {pt: colors[i] for i, pt in enumerate(main_bar_types) if pt in available_types}
    
    # Plot main bars
    x = np.arange(len(pivot_data.index))
    width = 0.18
    
    bar_positions = {}
    # Create readable labels for profit types
    readable_labels = {
        'base_case': 'Base Case',
        'max_prices': 'Max Prices', 
        'sol_real': 'Solution (No Trust Region) Real Profit',
        'sol_tr_real': 'Solution (Trust Region) Real Profit'
    }
    
    for i, profit_type in enumerate(available_types):
        if profit_type in pivot_data.columns:
            values = pivot_data[profit_type].values
            readable_label = readable_labels.get(profit_type, profit_type.replace('_', ' ').title())
            bars = ax.bar(x + i * width, values, width, label=readable_label, 
                         color=type_colors[profit_type], alpha=0.8)
            bar_positions[profit_type] = x + i * width
    
    # Add predicted values as horizontal lines over the corresponding real bars
    line_width = width * 0.8
    
    # Add predicted line for sol_real bars
    if 'sol_real' in pivot_data.columns and 'sol_predicted' in pivot_data.columns:
        sol_real_pos = bar_positions['sol_real']
        predicted_values = pivot_data['sol_predicted'].values
        for i, (pos, pred_val) in enumerate(zip(sol_real_pos, predicted_values)):
            if not np.isnan(pred_val):
                ax.plot([pos - line_width/2, pos + line_width/2], [pred_val, pred_val], 
                       color='lightblue', linewidth=4, alpha=0.9)
        # Add legend entry
        ax.plot([], [], color='lightblue', linewidth=4, label='Solution (No Trust Region) Predicted Profit', alpha=0.9)
    
    # Add predicted line for sol_tr_real bars
    if 'sol_tr_real' in pivot_data.columns and 'sol_tr_predicted' in pivot_data.columns:
        sol_tr_real_pos = bar_positions['sol_tr_real']
        tr_predicted_values = pivot_data['sol_tr_predicted'].values
        for i, (pos, pred_val) in enumerate(zip(sol_tr_real_pos, tr_predicted_values)):
            if not np.isnan(pred_val):
                ax.plot([pos - line_width/2, pos + line_width/2], [pred_val, pred_val], 
                       color='lightgreen', linewidth=4, alpha=0.9)
        # Add legend entry
        ax.plot([], [], color='lightgreen', linewidth=4, label='Solution (Trust Region) Predicted Profit', alpha=0.9)
    
    ax.set_xlabel('Number of Controlled Stations', fontsize=12)
    ax.set_ylabel('Average Profit ($)', fontsize=12)
    ax.set_title('Average Profit by Number of Controlled Stations', fontsize=14, fontweight='bold')
    ax.set_xticks(x + width * (len(available_types) - 1) / 2)
    ax.set_xticklabels(pivot_data.index)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plot1_file = os.path.join(output_dir, "aggregator_analysis_profit_by_stations.png")
    plt.savefig(plot1_file, dpi=300, bbox_inches='tight')
    print(f"  → Profit comparison plot saved to: {plot1_file}")
    plt.close()
    
    # 2. Save prediction accuracy plot
    if not pred_real_df.empty:
        fig2, ax = plt.subplots(figsize=(10, 8))
        
        # Use consistent colors for models
        model_colors = {
            'With Trust Region': 'green',
            'No Trust Region': 'blue'
        }
        for model_type in pred_real_df['model'].unique():
            model_data = pred_real_df[pred_real_df['model'] == model_type]
            color = model_colors.get(model_type, 'gray')
            ax.scatter(model_data['predicted'], model_data['real'], 
                       label=model_type, alpha=0.7, s=80, color=color)
        
        min_val = min(pred_real_df['predicted'].min(), pred_real_df['real'].min())
        max_val = max(pred_real_df['predicted'].max(), pred_real_df['real'].max())
        ax.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5, label='Perfect Prediction')
        
        ax.set_xlabel('Predicted Profit ($)', fontsize=12)
        ax.set_ylabel('Real Profit ($)', fontsize=12)
        ax.set_title('Prediction Accuracy: Predicted vs Real Profit', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plot2_file = os.path.join(output_dir, "aggregator_analysis_prediction_accuracy.png")
        plt.savefig(plot2_file, dpi=300, bbox_inches='tight')
        print(f"  → Prediction accuracy plot saved to: {plot2_file}")
        plt.close()
    
    # 3. Save trust region effectiveness plot
    if not tr_comp_df.empty:
        fig3, ax = plt.subplots(figsize=(10, 8))
        
        tr_by_size = tr_comp_df.groupby('num_controlled').agg({
            'with_tr': 'mean',
            'without_tr': 'mean',
            'improvement': 'mean'
        }).reset_index()
        
        x = np.arange(len(tr_by_size))
        width = 0.35
        
        ax.bar(x - width/2, tr_by_size['without_tr'], width, label='Without Trust Region', 
               color='blue', alpha=0.7)
        ax.bar(x + width/2, tr_by_size['with_tr'], width, label='With Trust Region', 
               color='green', alpha=0.7)
        
        ax.set_xlabel('Number of Controlled Stations', fontsize=12)
        ax.set_ylabel('Average Real Profit ($)', fontsize=12)
        ax.set_title('Trust Region Effectiveness', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(tr_by_size['num_controlled'])
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plot3_file = os.path.join(output_dir, "aggregator_analysis_trust_region_effectiveness.png")
        plt.savefig(plot3_file, dpi=300, bbox_inches='tight')
        print(f"  → Trust region effectiveness plot saved to: {plot3_file}")
        plt.close()
    
    # 4. Save improvement over baselines plot
    if not imp_df.empty:
        fig4, ax = plt.subplots(figsize=(12, 8))
        
        # Use consistent colors for methods
        method_colors = {
            'Aggregator (Trust Region)': 'green',
            'Aggregator (No Trust Region)': 'blue', 
            'Max Prices': 'red'
        }
        
        sns.boxplot(data=imp_df, x='method', y='improvement_pct', hue='method', ax=ax, 
                   palette=method_colors, legend=False)
        ax.set_xlabel('Method', fontsize=12)
        ax.set_ylabel('Improvement over Base Case (%)', fontsize=12)
        ax.set_title('Profit Improvement over Base Case', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='red', linestyle='--', alpha=0.5)
        
        plot4_file = os.path.join(output_dir, "aggregator_analysis_improvement_over_baseline.png")
        plt.savefig(plot4_file, dpi=300, bbox_inches='tight')
        print(f"  → Improvement over baseline plot saved to: {plot4_file}")
        plt.close()
    
    print(f"\nAll plots saved in directory: {output_dir}")
    
    return pred_real_df, tr_comp_df, imp_df


def print_summary_statistics(df, pred_real_df, tr_comp_df, imp_df):
    """Print summary statistics to answer the research questions."""
    print("\n" + "="*80)
    print("SUMMARY STATISTICS AND ANALYSIS")
    print("="*80)
    
    # Question 1: Prediction accuracy
    print("\n1. PREDICTION ACCURACY (Predicted vs Real Profit)")
    print("-" * 50)
    
    if not pred_real_df.empty:
        for model_type in pred_real_df['model'].unique():
            model_data = pred_real_df[pred_real_df['model'] == model_type]
            correlation = model_data['predicted'].corr(model_data['real'])
            mae = np.mean(np.abs(model_data['predicted'] - model_data['real']))
            rmse = np.sqrt(np.mean((model_data['predicted'] - model_data['real'])**2))
            
            print(f"{model_type}:")
            print(f"  Correlation: {correlation:.3f}")
            print(f"  Mean Absolute Error: ${mae:.2f}")
            print(f"  Root Mean Square Error: ${rmse:.2f}")
            print()
    
    # Question 2: Trust region effectiveness
    print("2. TRUST REGION EFFECTIVENESS")
    print("-" * 50)
    
    if not tr_comp_df.empty:
        tr_better = (tr_comp_df['improvement'] > 0).sum()
        tr_worse = (tr_comp_df['improvement'] < 0).sum()
        tr_same = (tr_comp_df['improvement'] == 0).sum()
        
        print(f"Trust Region performs better: {tr_better}/{len(tr_comp_df)} cases ({tr_better/len(tr_comp_df)*100:.1f}%)")
        print(f"Trust Region performs worse: {tr_worse}/{len(tr_comp_df)} cases ({tr_worse/len(tr_comp_df)*100:.1f}%)")
        print(f"Trust Region performs same: {tr_same}/{len(tr_comp_df)} cases ({tr_same/len(tr_comp_df)*100:.1f}%)")
        print(f"Average improvement with TR: ${tr_comp_df['improvement'].mean():.2f}")
        print()
    
    # Question 3: Comparison with baselines
    print("3. COMPARISON WITH BASELINES")
    print("-" * 50)
    
    if not imp_df.empty:
        for method in imp_df['method'].unique():
            method_data = imp_df[imp_df['method'] == method]
            better_cases = (method_data['improvement'] > 0).sum()
            worse_cases = (method_data['improvement'] < 0).sum()
            same_cases = (method_data['improvement'] == 0).sum()
            
            print(f"{method}:")
            print(f"  Better than base case: {better_cases}/{len(method_data)} cases ({better_cases/len(method_data)*100:.1f}%)")
            print(f"  Worse than base case: {worse_cases}/{len(method_data)} cases ({worse_cases/len(method_data)*100:.1f}%)")
            print(f"  Average improvement: {method_data['improvement_pct'].mean():.1f}%")
            print()
    
    # Question 4: Effect of number of controlled stations
    print("4. EFFECT OF NUMBER OF CONTROLLED STATIONS")
    print("-" * 50)
    
    if not imp_df.empty:
        by_size = imp_df.groupby(['num_controlled', 'method'])['improvement_pct'].mean().reset_index()
        
        for method in by_size['method'].unique():
            method_data = by_size[by_size['method'] == method]
            print(f"{method}:")
            for _, row in method_data.iterrows():
                print(f"  {int(row['num_controlled'])} stations: {row['improvement_pct']:.1f}% improvement")
            print()


def main():
    """Main function to run the complete analysis."""
    print("Aggregator Experiments Analysis")
    print("=" * 50)
    
    # Define the specific CSV files to analyze
    csv_files = [
        "../results/aggregator_experiments_20250614_150755.csv",  # 1-3 stations
        "../results/aggregator_experiments_20250614_164920.csv"   # 4-5 stations
    ]
    
    try:
        # Load and combine data
        print("Loading and combining data...")
        df = load_and_combine_results(csv_files)
        
        # Preprocess data
        print("Preprocessing data...")
        df = preprocess_data(df)
        
        print(f"Data overview:")
        print(f"  Total combinations: {df['controlled_stations'].nunique()}")
        print(f"  Number of controlled stations range: {df['num_controlled'].min()}-{df['num_controlled'].max()}")
        print(f"  Profit types: {df['type'].unique().tolist()}")
        
        # Create visualizations
        print("\nCreating visualizations...")
        pred_real_df, tr_comp_df, imp_df = create_comprehensive_analysis(df)
        
        # Print summary statistics
        print_summary_statistics(df, pred_real_df, tr_comp_df, imp_df)
        
        print("\nAnalysis completed successfully!")
        
    except Exception as e:
        print(f"Error during analysis: {e}")
        raise


if __name__ == "__main__":
    main() 