import pandas as pd
import numpy as np
import os
import pickle
import opticl
import itertools
from aggregator_model.get_aggregator_map_data import load_aggregator_excel_data
from sklearn.model_selection import train_test_split


def train_profit_regression_model(scenarios_file, demand_file, aggregator_excel_file, 
                                  output_folder, prefix, cv_folds=5, verbose=1):
    """
    Train regression models to predict profit based on charging prices.
    
    Parameters:
    -----------
    scenarios_file : str
        Path to CSV file containing charging prices for each scenario
    demand_file : str
        Path to CSV file containing aggregated demand data
    aggregator_excel_file : str
        Path to Excel file containing electricity costs
    output_folder : str
        Folder path to save results
    prefix : str
        Prefix for output filenames
    cv_folds : int, default=5
        Number of cross-validation folds for model training
    verbose : int
        Verbosity level (1=show info, 2=show detailed scenario-by-scenario info)
    """
    
    if verbose >= 1:
        print("Loading input data...")
    
    # Load charging prices (features)
    scenarios_df = pd.read_csv(scenarios_file)
    charging_stations = [col for col in scenarios_df.columns if col != 'scenario']
    if verbose >= 1:
        print(f"Loaded {len(scenarios_df)} scenarios for charging stations: {charging_stations}")
    
    # Load demand data
    demand_df = pd.read_csv(demand_file)
    if verbose >= 1:
        print(f"Loaded demand data: {len(demand_df)} records")
    
    # Load electricity costs
    aggregator_data = load_aggregator_excel_data(aggregator_excel_file, verbose=verbose)
    electricity_costs = aggregator_data[None]['pElectricityCost']  # C_t for each period t
    if verbose >= 1:
        print(f"Loaded electricity costs for {len(electricity_costs)} time periods")
    
    # Prepare training data
    if verbose >= 1:
        print("Preparing training data...")
    
    # Create features DataFrame (charging prices) - will be filtered after processing
    feature_columns = [f'rc_{station}' for station in charging_stations]
    
    # Get scenarios that exist in both files
    available_scenarios_in_demand = set(demand_df['scenario'].unique())
    available_scenarios_in_prices = set(scenarios_df['scenario'].unique())
    common_scenarios = available_scenarios_in_demand.intersection(available_scenarios_in_prices)
    
    if verbose >= 1:
        print(f"Scenarios in prices file: {len(available_scenarios_in_prices)}")
        print(f"Scenarios in demand file: {len(available_scenarios_in_demand)}")
        print(f"Common scenarios to process: {len(common_scenarios)}")
        if len(common_scenarios) < len(available_scenarios_in_prices):
            missing_scenarios = available_scenarios_in_prices - available_scenarios_in_demand
            print(f"Scenarios missing demand data (will be skipped): {len(missing_scenarios)}")
    
    # Calculate profit for each scenario (target variable)
    profits = []
    processed_scenarios = []
    
    for scenario in sorted(common_scenarios):
        if verbose >= 1 and scenario % 1000 == 0:
            print(f"Processing scenario {scenario}...")
        
        # Get charging prices for this scenario
        scenario_prices = scenarios_df[scenarios_df['scenario'] == scenario]
        prices = {station: scenario_prices[str(station)].iloc[0] for station in charging_stations}
        
        # Get demand data for this scenario
        scenario_demand = demand_df[demand_df['scenario'] == scenario]
        
        if verbose >= 2:
            print(f"\n--- Scenario {scenario} Details ---")
            print(f"Charging prices: {prices}")
            print(f"Total demand records: {len(scenario_demand)}")
        
        # Calculate revenue: sum_i sum_t r^C_i * d_{i,t}
        revenue = 0
        revenue_details = {}
        for _, row in scenario_demand.iterrows():
            station = str(int(row['charging_station']))  # Convert to string to match prices keys
            demand = row['aggregated_demand']
            price = prices[station]
            station_revenue = price * demand
            revenue += station_revenue
            
            if verbose >= 2 and demand > 0:  # Only show non-zero demand
                if station not in revenue_details:
                    revenue_details[station] = []
                revenue_details[station].append(f"t{row['time_period']}:{demand:.3f}kWh*${price:.3f}=${station_revenue:.4f}")
        
        # Calculate cost: sum_i sum_t C_t * d_{i,t}
        cost = 0
        cost_details = {}
        for _, row in scenario_demand.iterrows():
            time_period = int(row['time_period'])  # Convert to int to match electricity_costs keys
            demand = row['aggregated_demand']
            elec_cost = electricity_costs[time_period]
            period_cost = elec_cost * demand
            cost += period_cost
            
            if verbose >= 2 and demand > 0:  # Only show non-zero demand
                station = str(int(row['charging_station']))  # Convert to string for consistency
                key = f"t{time_period}"
                if key not in cost_details:
                    cost_details[key] = []
                cost_details[key].append(f"s{station}:{demand:.3f}kWh*${elec_cost:.3f}=${period_cost:.4f}")
        
        # Profit = revenue - cost
        profit = revenue - cost
        profits.append(profit)
        processed_scenarios.append(scenario)
        
        if verbose >= 2:
            print(f"Revenue breakdown by station:")
            for station, details in revenue_details.items():
                print(f"  Station {station}: {' + '.join(details)}")
            print(f"Cost breakdown by time period:")
            for period, details in cost_details.items():
                print(f"  {period}: {' + '.join(details)}")
            print(f"Total Revenue: ${revenue:.4f}")
            print(f"Total Cost: ${cost:.4f}")
            print(f"Profit: ${profit:.4f}")
            print(f"--- End Scenario {scenario} ---\n")
    
    # Create features DataFrame for processed scenarios only
    X = scenarios_df[scenarios_df['scenario'].isin(processed_scenarios)][charging_stations].copy()
    X.columns = feature_columns
    X = X.reset_index(drop=True)  # Reset index to match profits array
    
    # Create target DataFrame
    y = pd.Series(profits, name='profit')
    
    if verbose >= 1:
        print(f"Training data prepared:")
        print(f"Processed scenarios: {len(processed_scenarios)}")
        print(f"Features shape: {X.shape}")
        print(f"Target shape: {y.shape}")
        print(f"Profit range: [{y.min():.4f}, {y.max():.4f}]")
        print(f"Feature columns: {list(X.columns)}")
        
        # Verify data consistency
        if len(X) != len(y):
            raise ValueError(f"Mismatch: Features have {len(X)} samples but targets have {len(y)} samples")
    
    # Create output directory
    os.makedirs(output_folder, exist_ok=True)
    
    # Save training data
    training_data = X.copy()
    training_data['profit'] = y
    training_data_file = os.path.join(output_folder, f"{prefix}_training_data.csv")
    training_data.to_csv(training_data_file, index=False)
    
    if verbose >= 1:
        print(f"Saved training data to: {training_data_file}")
    
    # Train multiple regression models
    alg_list = ['linear', 'rf', 'svm', 'cart', 'gbm', 'mlp']
    task_type = 'continuous'
    outcome = 'profit'
    seed = 42
    if verbose >= 1:
        print(f"Training regression models: {alg_list}")
    
    # Handle small datasets - adjust cv_folds and train/test split
    n_samples = len(X)
    if n_samples < 10:  # Very small dataset
        if verbose >= 1:
            print(f"Small dataset detected ({n_samples} samples). Using all data for training (no test split).")
        X_train, X_test = X, X  # Use all data for both training and testing
        y_train, y_test = y, y
        # Adjust cv_folds to be at most n_samples, minimum 2 (required by scikit-learn)
        cv_folds_adjusted = max(2, min(cv_folds, n_samples))
        if cv_folds_adjusted != cv_folds:
            if verbose >= 1:
                print(f"Adjusted cv_folds from {cv_folds} to {cv_folds_adjusted} due to small sample size.")
    else:
        # Normal train/test split for larger datasets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=seed
        )
        cv_folds_adjusted = cv_folds
    
    # Ensure feature names are preserved in DataFrames
    if not isinstance(X_train, pd.DataFrame):
        X_train = pd.DataFrame(X_train, columns=feature_columns)
    if not isinstance(X_test, pd.DataFrame):
        X_test = pd.DataFrame(X_test, columns=feature_columns)
    if not isinstance(y_train, pd.Series):
        y_train = pd.Series(y_train, name='profit')
    if not isinstance(y_test, pd.Series):
        y_test = pd.Series(y_test, name='profit')
    
    all_performance = []
    
    # Train models for each algorithm
    for alg in alg_list:
        if verbose >= 1:
            print(f"Training {alg} model...")
        
        # Create algorithm-specific directory (required by opticl)
        alg_dir = os.path.join(output_folder, alg)
        os.makedirs(alg_dir, exist_ok=True)
        
        # Map algorithm names to opticl names
        alg_run = 'rf_shallow' if alg == 'rf' else alg
        
        try:
            # Train model
            model_save = os.path.join(alg_dir, f"{prefix}_{outcome}_model.csv")
            m, perf = opticl.run_model(X_train, y_train, X_test, y_test, alg_run, outcome,
                                       task=task_type, 
                                       seed=seed, 
                                       cv_folds=cv_folds_adjusted, 
                                       parameter_grid=None,
                                       save_path=model_save,
                                       save=False,
                                       save_pickle=False,  # We'll save pickle manually with prefix
                                       pickle_dir=alg_dir)
            
            # Save pickle file with prefix in filename
            pickle_file = os.path.join(alg_dir, f"{prefix}_{outcome}_{alg}_model.pkl")
            with open(pickle_file, 'wb') as f:
                pickle.dump(m, f)
            
            # Save model for ConstraintLearning class (required for opticl optimization)
            constraintL = opticl.ConstraintLearning(X_train, pd.DataFrame(y_train), m, alg)
            constraint_add = constraintL.constraint_extrapolation(task_type)
            constraint_add.to_csv(model_save, index=False)
            
            # Save performance
            perf['seed'] = seed
            perf['outcome'] = outcome
            perf['alg'] = alg
            perf_save = os.path.join(alg_dir, f"{prefix}_{outcome}_performance.csv")
            perf.to_csv(perf_save, index=False)
            
            # Store for comparison
            all_performance.append(perf)
            
            if verbose >= 1:
                print(f"{alg} - R² score: {perf.iloc[0]['test_r2']:.4f}")
            
        except Exception as e:
            if verbose >= 1:
                print(f"Failed to train {alg}: {str(e)}")
            continue
    
    # Create comprehensive performance comparison
    if all_performance:
        performance_df = pd.concat(all_performance, ignore_index=True)
        performance_file = os.path.join(output_folder, f"{prefix}_performance_comparison.csv")
        performance_df.to_csv(performance_file, index=False)
        
        if verbose >= 1:
            print(f"\nSaved performance comparison to: {performance_file}")
            print(f"\n{'='*60}")
            print("PERFORMANCE COMPARISON")
            print(f"{'='*60}")
            print(f"{'Algorithm':<10} {'R² Score':<12} {'Valid Score':<12} {'Test Score':<12}")
            print(f"{'-'*60}")
            
            for _, row in performance_df.iterrows():
                print(f"{row['alg']:<10} {row['test_r2']:<12.4f} {row['valid_score']:<12.4f} {row['test_score']:<12.4f}")
            
            # Find best performing algorithm
            best_alg_row = performance_df.loc[performance_df['test_r2'].idxmax()]
            best_alg = best_alg_row['alg']
            print(f"\nBest performing algorithm: {best_alg} (R² = {best_alg_row['test_r2']:.4f})")
            
            # Save best model info for model selection
            best_model_file = os.path.join(output_folder, best_alg, f"{prefix}_{outcome}_model.csv")
            print(f"Best model saved at: {best_model_file}")
    
    if verbose >= 1:
        print("\nTraining completed successfully!")
        print(f"All results saved in: {output_folder}")
        print(f"Files created:")
        for filename in os.listdir(output_folder):
            if filename.startswith(prefix):
                print(f"  - {filename}")
