import pandas as pd
import numpy as np
import os
import pickle
import opticl
import itertools
from aggregator_model import load_aggregator_excel_data
from sklearn.model_selection import train_test_split
from .compute_profit import compute_profit_stations


def train_competition_regression_models(scenarios_file, demand_file, aggregator_excel_file, 
                                      output_folder, prefix, cv_folds=5, verbose=1):
    """
    Train regression models to predict profit for each charging station separately.
    
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
    
    # Calculate profit for each station in each scenario (target variables)
    station_profits_dict = {station: [] for station in charging_stations}
    processed_scenarios = []
    
    for scenario in sorted(common_scenarios):
        if verbose >= 1 and scenario % 1000 == 0:
            print(f"Processing scenario {scenario}...")
        
        try:
            # Use the compute_profit_stations function to get profit for each station
            station_profits = compute_profit_stations(
                scenario=scenario,
                demand_df=demand_df,
                scenarios_df=scenarios_df,
                electricity_costs=electricity_costs,
                verbose=verbose if verbose >= 2 else 0  # Only show detailed computations if verbose >= 2
            )
            
            # Store profits for each station
            for station in charging_stations:
                station_id = str(station)
                profit = station_profits.get(station_id, 0.0)  # Default to 0 if station not in results
                station_profits_dict[station].append(profit)
            
            processed_scenarios.append(scenario)
            
        except Exception as e:
            if verbose >= 1:
                print(f"Error processing scenario {scenario}: {e}")
            continue
    
    # Create features DataFrame for processed scenarios only
    X = scenarios_df[scenarios_df['scenario'].isin(processed_scenarios)][charging_stations].copy()
    X.columns = feature_columns
    X = X.reset_index(drop=True)  # Reset index to match profits array
    
    # Create target DataFrames for each station
    y_dict = {}
    outcome_list = []
    for station in charging_stations:
        outcome_name = f'profit_{station}'
        outcome_list.append(outcome_name)
        y_dict[outcome_name] = pd.Series(station_profits_dict[station], name=outcome_name)
    
    if verbose >= 1:
        print(f"Training data prepared:")
        print(f"Processed scenarios: {len(processed_scenarios)}")
        print(f"Features shape: {X.shape}")
        print(f"Outcomes: {outcome_list}")
        print(f"Feature columns: {list(X.columns)}")
        
        # Show profit ranges for each station
        for outcome_name, y_series in y_dict.items():
            print(f"{outcome_name} range: [{y_series.min():.4f}, {y_series.max():.4f}]")
        
        # Verify data consistency
        for outcome_name, y_series in y_dict.items():
            if len(X) != len(y_series):
                raise ValueError(f"Mismatch: Features have {len(X)} samples but {outcome_name} has {len(y_series)} samples")
    
    # Create output directory
    os.makedirs(output_folder, exist_ok=True)
    
    # Save training data
    training_data = X.copy()
    for outcome_name, y_series in y_dict.items():
        training_data[outcome_name] = y_series
    training_data_file = os.path.join(output_folder, f"{prefix}_training_data.csv")
    training_data.to_csv(training_data_file, index=False)
    
    if verbose >= 1:
        print(f"Saved training data to: {training_data_file}")
    
    # Train multiple regression models for each outcome
    alg_list = ['linear', 'rf', 'svm', 'cart', 'gbm', 'mlp']
    task_type = 'continuous'
    seed = 42
    if verbose >= 1:
        print(f"Training regression models: {alg_list}")
        print(f"Outcomes to train: {outcome_list}")
    
    # Handle small datasets - adjust cv_folds and train/test split
    n_samples = len(X)
    if n_samples < 10:  # Very small dataset
        if verbose >= 1:
            print(f"Small dataset detected ({n_samples} samples). Using all data for training (no test split).")
        X_train, X_test = X, X  # Use all data for both training and testing
        y_train_dict = {outcome: y_dict[outcome] for outcome in outcome_list}
        y_test_dict = {outcome: y_dict[outcome] for outcome in outcome_list}
        # Adjust cv_folds to be at most n_samples, minimum 2 (required by scikit-learn)
        cv_folds_adjusted = max(2, min(cv_folds, n_samples))
        if cv_folds_adjusted != cv_folds:
            if verbose >= 1:
                print(f"Adjusted cv_folds from {cv_folds} to {cv_folds_adjusted} due to small sample size.")
    else:
        # Normal train/test split for larger datasets
        X_train, X_test = train_test_split(X, test_size=0.2, random_state=seed)
        y_train_dict = {}
        y_test_dict = {}
        for outcome in outcome_list:
            y_train_dict[outcome], y_test_dict[outcome] = train_test_split(
                y_dict[outcome], test_size=0.2, random_state=seed
            )
        cv_folds_adjusted = cv_folds
    
    # Ensure feature names are preserved in DataFrames
    if not isinstance(X_train, pd.DataFrame):
        X_train = pd.DataFrame(X_train, columns=feature_columns)
    if not isinstance(X_test, pd.DataFrame):
        X_test = pd.DataFrame(X_test, columns=feature_columns)
    
    # Ensure target variables are pandas Series
    for outcome in outcome_list:
        if not isinstance(y_train_dict[outcome], pd.Series):
            y_train_dict[outcome] = pd.Series(y_train_dict[outcome], name=outcome)
        if not isinstance(y_test_dict[outcome], pd.Series):
            y_test_dict[outcome] = pd.Series(y_test_dict[outcome], name=outcome)
    
    all_performance = []
    
    # Train models for each outcome and algorithm pair (similar to OptiCL reference)
    for outcome in outcome_list:
        if verbose >= 1:
            print(f"\nTraining models for outcome: {outcome}")
        
        for alg in alg_list:
            if verbose >= 1:
                print(f"  Training {alg} model for {outcome}...")
            
            # Create algorithm-specific directory (required by opticl)
            alg_dir = os.path.join(output_folder, alg)
            os.makedirs(alg_dir, exist_ok=True)
            
            # Map algorithm names to opticl names
            alg_run = 'rf_shallow' if alg == 'rf' else alg
            
            try:
                # Train model
                model_save = os.path.join(alg_dir, f"{prefix}_{outcome}_model.csv")
                m, perf = opticl.run_model(X_train, y_train_dict[outcome], X_test, y_test_dict[outcome], alg_run, outcome,
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
                constraintL = opticl.ConstraintLearning(X_train, pd.DataFrame(y_train_dict[outcome]), m, alg)
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
                    print(f"    {alg} - R² score: {perf.iloc[0]['test_r2']:.4f}")
                
            except Exception as e:
                if verbose >= 1:
                    print(f"    Failed to train {alg} for {outcome}: {str(e)}")
                continue
    
    # Create comprehensive performance comparison
    if all_performance:
        performance_df = pd.concat(all_performance, ignore_index=True)
        performance_file = os.path.join(output_folder, f"{prefix}_performance_comparison.csv")
        performance_df.to_csv(performance_file, index=False)
        
        if verbose >= 1:
            print(f"\nSaved performance comparison to: {performance_file}")
            print(f"\n{'='*80}")
            print("PERFORMANCE COMPARISON")
            print(f"{'='*80}")
            print(f"{'Outcome':<15} {'Algorithm':<10} {'R² Score':<12} {'Valid Score':<12} {'Test Score':<12}")
            print(f"{'-'*80}")
            
            for _, row in performance_df.iterrows():
                print(f"{row['outcome']:<15} {row['alg']:<10} {row['test_r2']:<12.4f} {row['valid_score']:<12.4f} {row['test_score']:<12.4f}")
            
            # Find best performing algorithm for each outcome
            print(f"\n{'='*60}")
            print("BEST PERFORMING ALGORITHMS BY OUTCOME")
            print(f"{'='*60}")
            for outcome in outcome_list:
                outcome_perf = performance_df[performance_df['outcome'] == outcome]
                if not outcome_perf.empty:
                    best_alg_row = outcome_perf.loc[outcome_perf['test_r2'].idxmax()]
                    best_alg = best_alg_row['alg']
                    print(f"{outcome:<15}: {best_alg} (R² = {best_alg_row['test_r2']:.4f})")
                    
                    # Save best model info for model selection
                    best_model_file = os.path.join(output_folder, best_alg, f"{prefix}_{outcome}_model.csv")
                    if verbose >= 2:
                        print(f"  Best model saved at: {best_model_file}")
    
    if verbose >= 1:
        print("\nTraining completed successfully!")
        print(f"All results saved in: {output_folder}")
        print(f"Trained models for {len(outcome_list)} outcomes: {outcome_list}")
        if verbose >= 2:
            print(f"Files created:")
            for filename in os.listdir(output_folder):
                if filename.startswith(prefix):
                    print(f"  - {filename}")
