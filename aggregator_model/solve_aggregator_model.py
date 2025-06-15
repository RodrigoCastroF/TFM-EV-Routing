"""
This function solves the simplified aggregator model with embedded OptiCL regression
"""

from .get_aggregator_map_data import load_aggregator_excel_data
from .save_aggregator_solution_data import extract_aggregator_solution_data, save_aggregator_solution_data
import pyomo.environ as pyo
from pyomo.opt import SolverFactory
import pandas as pd
import opticl
import numpy as np


def solve_aggregator_model(input_excel_file=None, input_data=None, performance_csv_file=None, training_data_csv_file=None, trust_region=True,
                          output_excel_file=None, model="auto", alg=None, solver="gurobi", time_limit=300, verbose=1):
    """
    Solve the aggregator optimization problem with embedded regression model.
    Automatically detects whether to use monopoly or competition model based on input data.

    Parameters
    ----------
    input_excel_file: str, optional
        Path to the input Excel file containing aggregator data (charging station bounds).
        Either this or input_data must be provided.
    input_data: dict, optional
        Input data in the format returned by load_aggregator_excel_data.
        Either this or input_excel_file must be provided.
    performance_csv_file: str
        Path to the CSV file containing performance comparison of regression models.
    training_data_csv_file: str
        Path to the CSV file containing training data for the regression models.
    trust_region: bool
        If True, solutions are restricted to lie in the trust region (the domain of the training data)
    output_excel_file: str, optional
        Path to save the solution Excel file (optional).
    model: str
        Model to use (default: "auto"). Options: "auto", "monopoly", "competition".
        "auto" will detect whether to use monopoly or competition model based on input data.
    alg: str, optional
        Algorithm to use for regression models (default: None for automatic selection).
        When None, the best algorithm is selected automatically.
        When specified (e.g., "mlp", "linear"), the provided algorithm is used for all regression models.
    solver: str
        Solver to use (default: "gurobi").
    time_limit: int
        Time limit in seconds (default: 300).
    verbose: int
        Verbosity level (0=silent, 1=basic, 2=detailed).

    Returns
    -------
    dict
        Dictionary with solution results including objective value and solution data.
    """
    
    # Load or use provided data
    if input_data is not None:
        if verbose >= 1:
            print(f"Using provided input data...")
    elif input_excel_file is not None:
        if verbose >= 1:
            print(f"Loading aggregator data from {input_excel_file}...")
        input_data = load_aggregator_excel_data(input_excel_file, verbose=verbose)
    else:
        raise ValueError("Either input_excel_file or input_data must be provided")
    
    # Check if this is a competition model (has fixed competitor prices)
    fixed_prices = input_data[None].get('pChargingPrice', {})
    has_competitors = any(not pd.isna(price) for price in fixed_prices.values())
    
    if model == "auto":
        if has_competitors:
            if verbose >= 1:
                print("Detected competition model (some stations have fixed prices)")
            model = "competition"
        else:
            if verbose >= 1:
                print("Detected monopoly model (all stations are optimizable)")
            model = "monopoly"

    if verbose >= 1:
        print(f"Using {model} model...")
    
    if model == "competition":
        return solve_competition_model(input_data=input_data, performance_csv_file=performance_csv_file, 
                                     training_data_csv_file=training_data_csv_file, 
                                     trust_region=trust_region, output_excel_file=output_excel_file, 
                                     alg=alg, solver=solver, time_limit=time_limit, verbose=verbose)
    else:
        return solve_monopoly_model(input_data=input_data, performance_csv_file=performance_csv_file, 
                                  training_data_csv_file=training_data_csv_file, 
                                  trust_region=trust_region, output_excel_file=output_excel_file, 
                                  alg=alg, solver=solver, time_limit=time_limit, verbose=verbose)


def solve_monopoly_model(input_data, performance_csv_file, training_data_csv_file, trust_region=True,
                        output_excel_file=None, alg=None, solver="gurobi", time_limit=300, verbose=1):
    """
    Solve the monopoly aggregator optimization problem with embedded regression model.
    (Original model that optimizes all stations)

    Parameters
    ----------
    input_data: dict
        Input data in the format returned by load_aggregator_excel_data.
    performance_csv_file: str
        Path to the CSV file containing performance comparison of regression models.
    training_data_csv_file: str
        Path to the CSV file containing training data for the regression models.
    trust_region: bool
        If True, solutions are restricted to lie in the trust region (the domain of the training data)
    output_excel_file: str, optional
        Path to save the solution Excel file (optional).
    alg: str, optional
        Algorithm to use for regression model (default: None for automatic selection).
    solver: str
        Solver to use (default: "gurobi").
    time_limit: int
        Time limit in seconds (default: 300).
    verbose: int
        Verbosity level (0=silent, 1=basic, 2=detailed).

    Returns
    -------
    dict
        Dictionary with solution results including objective value and solution data.
    """

    charging_stations = input_data[None]['sChargingStations'][None]
    min_prices = input_data[None]['pMinChargingPrice']
    max_prices = input_data[None]['pMaxChargingPrice']
    
    if verbose >= 1:
        print(f"Loaded {len(charging_stations)} charging stations: {charging_stations}")

    # Load regression model data
    if verbose >= 1:
        print(f"Loading regression model data...")
    performance_df = pd.read_csv(performance_csv_file)
    training_data = pd.read_csv(training_data_csv_file)
    feature_columns = [col for col in training_data.columns if col != 'profit']
    X_train = training_data[feature_columns]
    
    if verbose >= 1:
        print(f"Available algorithms: {performance_df['alg'].tolist()}")
        print(f"Feature columns: {feature_columns}")

    # Select regression model
    if alg is not None:
        # Use specified algorithm
        alg_rows = performance_df[performance_df['alg'] == alg]
        if alg_rows.empty:
            raise ValueError(f"Algorithm '{alg}' not found in performance data. Available algorithms: {performance_df['alg'].unique().tolist()}")
        best_row = alg_rows.iloc[0]  # Take first row if multiple exist
        best_alg = alg
        if verbose >= 1:
            print(f"Using specified algorithm: {best_alg} (R² = {best_row['test_r2']:.4f})")
    else:
        # Select best algorithm automatically
        best_row = performance_df.loc[performance_df['test_r2'].idxmax()]
        best_alg = best_row['alg']
        if verbose >= 1:
            print(f"Selected best algorithm: {best_alg} (R² = {best_row['test_r2']:.4f})")

    # Prepare OptiCL model selection
    performance_for_selection = pd.DataFrame([{
        'alg': best_alg,
        'outcome': 'profit',
        'valid_score': best_row['valid_score'],
        'save_path': best_row['save_path'],
        'task': 'continuous',
        'seed': best_row['seed']
    }])
    
    model_master = opticl.model_selection(performance_for_selection, 
                                          constraints_embed=[], 
                                          objectives_embed={'profit': 1})
    model_master['lb'] = None
    model_master['ub'] = None  
    model_master['SCM_counterfactuals'] = None
    model_master['features'] = [feature_columns] * len(model_master.index)
    if verbose >= 1:
        print("\nModel master:")
        print(model_master.to_string())

    # Create concrete model from scratch following OptiCL pattern
    if verbose >= 1:
        print("\nCreating concrete model...")
    
    model = pyo.ConcreteModel()
    
    # Create variable names and bounds
    var_names = [f'rc_{station}' for station in charging_stations]
    bounds_dict = {}
    for station in charging_stations:
        var_name = f'rc_{station}'
        min_price = min_prices[station]
        max_price = max_prices[station]
        bounds_dict[var_name] = (min_price, max_price)
        if verbose >= 2:
            print(f"Variable {var_name} will have bounds [{min_price}, {max_price}]")

    # Create indexed variable following OptiCL pattern
    model.x = pyo.Var(var_names, domain=pyo.Reals, 
                      bounds=lambda m, var_name: bounds_dict[var_name])

    # Placeholder objective (will be replaced by OptiCL)
    model.OBJ = pyo.Objective(expr=0, sense=pyo.maximize)

    # Embed regression model using OptiCL
    if verbose >= 1:
        print("Embedding regression model with OptiCL...")
    
    final_model = opticl.optimization_MIP(model, model.x, model_master, X_train, tr=trust_region)
    
    if verbose >= 1:
        print(f"Model created with {len(list(final_model.component_objects(pyo.Constraint)))} constraints")

    # Solve the model
    if verbose >= 1:
        print(f"Solving with {solver}...")
    
    opt = SolverFactory(solver)
    time_limit_option = {"cbc": "seconds", "gurobi": "timeLimit", "glpk": "tmlim", "cplex": "timelimit"}
    if solver in time_limit_option:
        opt.options[time_limit_option[solver]] = time_limit

    results = opt.solve(final_model, tee=(verbose >= 2))

    # Process results
    if results.solver.status not in [pyo.SolverStatus.ok, pyo.SolverStatus.aborted]:
        if verbose >= 1:
            print(f"Solver failed: {results.solver.status}")
        return {'solver_status': 'failed'}

    if verbose >= 1:
        print(f"Solver status: {results.solver.status}")
        print(f"Termination: {results.solver.termination_condition}")

    # Extract results
    obj_value = pyo.value(final_model.OBJ)
    charging_prices = {}
    
    for station in charging_stations:
        var_name = f'rc_{station}'
        try:
            price = pyo.value(final_model.x[var_name])
            charging_prices[station] = price
        except (ValueError, AttributeError) as e:
            charging_prices[station] = None

    if verbose >= 1:
        print(f"Predicted profit: ${obj_value:.2f}")
        print("Optimal charging prices:")
        for station, price in charging_prices.items():
            min_p = min_prices[station]
            max_p = max_prices[station]
            if price is not None:
                print(f"  Station {station}: ${price:.3f}/kWh (range: ${min_p:.1f}-${max_p:.1f})")
            else:
                print(f"  Station {station}: UNKNOWN/NOT SET (range: ${min_p:.1f}-${max_p:.1f})")

    # Extract solution data directly from the final model
    if verbose >= 1:
        print("Extracting solution data...")
    
    solution_data = extract_aggregator_solution_data(final_model, charging_stations, min_prices, max_prices)
    
    if output_excel_file:
        if verbose >= 1:
            print(f"Saving solution to {output_excel_file}...")
        save_aggregator_solution_data(solution_data, output_excel_file)

    return {
        'solver_status': 'optimal',
        'objective_value': obj_value,
        'execution_time': results.solver.time if hasattr(results.solver, 'time') else None,
        'solution_data': solution_data,
        'charging_prices': charging_prices
    }


def solve_competition_model(input_data, performance_csv_file, training_data_csv_file, trust_region=True,
                           output_excel_file=None, alg=None, solver="gurobi", time_limit=300, verbose=1):
    """
    Solve the competition aggregator optimization problem with embedded regression models.
    (New model that optimizes only aggregator-controlled stations against fixed competitor prices)

    Parameters
    ----------
    input_data: dict
        Input data in the format returned by load_aggregator_excel_data.
    performance_csv_file: str
        Path to the CSV file containing performance comparison of regression models for each station.
    training_data_csv_file: str
        Path to the CSV file containing training data for the regression models.
    trust_region: bool
        If True, solutions are restricted to lie in the trust region (the domain of the training data)
    output_excel_file: str, optional
        Path to save the solution Excel file (optional).
    alg: str, optional
        Algorithm to use for regression models (default: None for automatic selection).
    solver: str
        Solver to use (default: "gurobi").
    time_limit: int
        Time limit in seconds (default: 300).
    verbose: int
        Verbosity level (0=silent, 1=basic, 2=detailed).

    Returns
    -------
    dict
        Dictionary with solution results including objective value and solution data.
    """

    all_stations = input_data[None]['sChargingStations'][None]
    min_prices = input_data[None]['pMinChargingPrice']
    max_prices = input_data[None]['pMaxChargingPrice']
    fixed_prices = input_data[None]['pChargingPrice']
    
    # Identify aggregator-controlled stations (have min/max bounds)
    aggregator_stations = [s for s in all_stations if not pd.isna(min_prices[s])]
    competitor_stations = [s for s in all_stations if not pd.isna(fixed_prices[s])]
    
    if verbose >= 1:
        print(f"Total stations: {len(all_stations)}")
        print(f"Aggregator-controlled stations: {aggregator_stations}")
        print(f"Competitor stations: {competitor_stations} with fixed prices: {[fixed_prices[s] for s in competitor_stations]}")

    # Load regression model data
    if verbose >= 1:
        print(f"Loading regression model data...")
    performance_df = pd.read_csv(performance_csv_file)
    training_data = pd.read_csv(training_data_csv_file)
    
    # Get all station-specific profit columns
    profit_columns = [col for col in training_data.columns if col.startswith('profit_')]
    feature_columns = [col for col in training_data.columns if not col.startswith('profit_')]
    X_train = training_data[feature_columns]
    
    if verbose >= 1:
        print(f"Feature columns: {feature_columns}")
        print(f"Profit columns: {profit_columns}")

    # Select regression model for each aggregator station
    station_models = {}
    performance_list = []
    
    for station in aggregator_stations:
        profit_col = f'profit_{station}'
        if profit_col not in profit_columns:
            raise ValueError(f"No profit column found for station {station}")
        
        # Filter performance data for this outcome
        station_perf = performance_df[performance_df['outcome'] == profit_col]
        if station_perf.empty:
            raise ValueError(f"No performance data found for outcome {profit_col}")
        
        # Select model for this station
        if alg is not None:
            # Use specified algorithm
            alg_rows = station_perf[station_perf['alg'] == alg]
            if alg_rows.empty:
                raise ValueError(f"Algorithm '{alg}' not found for station {station}. Available algorithms: {station_perf['alg'].unique().tolist()}")
            best_row = alg_rows.iloc[0]  # Take first row if multiple exist
            if verbose >= 1:
                print(f"Station {station}: Using specified algorithm {alg} (R² = {best_row['test_r2']:.4f})")
        else:
            # Select best algorithm automatically
            best_row = station_perf.loc[station_perf['test_r2'].idxmax()]
            if verbose >= 1:
                print(f"Station {station}: Best algorithm {best_row['alg']} (R² = {best_row['test_r2']:.4f})")
        
        station_models[station] = best_row
        
        # Add to performance list for OptiCL
        performance_list.append({
            'alg': best_row['alg'],
            'outcome': profit_col,
            'valid_score': best_row['valid_score'],
            'save_path': best_row['save_path'],
            'task': 'continuous',
            'seed': best_row['seed']
        })

    # Prepare OptiCL model selection - only include aggregator stations in objective
    objectives_embed = {f'profit_{station}': 1 for station in aggregator_stations}
    performance_for_selection = pd.DataFrame(performance_list)
    
    model_master = opticl.model_selection(performance_for_selection, 
                                          constraints_embed=[], 
                                          objectives_embed=objectives_embed)
    model_master['lb'] = None
    model_master['ub'] = None  
    model_master['SCM_counterfactuals'] = None
    model_master['features'] = [feature_columns] * len(model_master.index)
    
    if verbose >= 1:
        print("\nModel master:")
        print(model_master.to_string())

    # Create concrete model
    if verbose >= 1:
        print("\nCreating concrete model...")
    
    model = pyo.ConcreteModel()
    
    # Create variables for ALL stations (both aggregator and competitor)
    # This is needed because the regression models expect features for all stations
    var_names = [f'rc_{station}' for station in all_stations]
    bounds_dict = {}
    
    for station in all_stations:
        var_name = f'rc_{station}'
        if station in aggregator_stations:
            # Variable station - use min/max bounds
            min_price = min_prices[station]
            max_price = max_prices[station]
            bounds_dict[var_name] = (min_price, max_price)
            if verbose >= 2:
                print(f"Variable {var_name} bounds: [{min_price}, {max_price}]")
        else:
            # Fixed competitor station - set as fixed value
            fixed_price = fixed_prices[station]
            bounds_dict[var_name] = (fixed_price, fixed_price)
            if verbose >= 2:
                print(f"Fixed {var_name} = {fixed_price}")

    # Create indexed variable
    model.x = pyo.Var(var_names, domain=pyo.Reals, 
                      bounds=lambda m, var_name: bounds_dict[var_name])

    # Placeholder objective (will be replaced by OptiCL)
    model.OBJ = pyo.Objective(expr=0, sense=pyo.maximize)

    # Embed regression models using OptiCL
    if verbose >= 1:
        print("Embedding regression models with OptiCL...")
    
    final_model = opticl.optimization_MIP(model, model.x, model_master, X_train, tr=trust_region)
    
    if verbose >= 1:
        print(f"Model created with {len(list(final_model.component_objects(pyo.Constraint)))} constraints")

    # Solve the model
    if verbose >= 1:
        print(f"Solving with {solver}...")
    
    opt = SolverFactory(solver)
    time_limit_option = {"cbc": "seconds", "gurobi": "timeLimit", "glpk": "tmlim", "cplex": "timelimit"}
    if solver in time_limit_option:
        opt.options[time_limit_option[solver]] = time_limit

    results = opt.solve(final_model, tee=(verbose >= 2))

    # Process results
    if results.solver.status not in [pyo.SolverStatus.ok, pyo.SolverStatus.aborted]:
        if verbose >= 1:
            print(f"Solver failed: {results.solver.status}")
        return {'solver_status': 'failed'}

    if verbose >= 1:
        print(f"Solver status: {results.solver.status}")
        print(f"Termination: {results.solver.termination_condition}")

    # Extract results
    obj_value = pyo.value(final_model.OBJ)
    charging_prices = {}
    
    for station in all_stations:
        var_name = f'rc_{station}'
        try:
            price = pyo.value(final_model.x[var_name])
            charging_prices[station] = price
        except (ValueError, AttributeError) as e:
            charging_prices[station] = None

    if verbose >= 1:
        print(f"Predicted aggregator profit: ${obj_value:.2f}")
        print("Charging prices:")
        for station in all_stations:
            price = charging_prices[station]
            if station in aggregator_stations:
                min_p = min_prices[station]
                max_p = max_prices[station]
                if price is not None:
                    print(f"  Station {station} (AGGREGATOR): ${price:.3f}/kWh (range: ${min_p:.1f}-${max_p:.1f})")
                else:
                    print(f"  Station {station} (AGGREGATOR): UNKNOWN/NOT SET (range: ${min_p:.1f}-${max_p:.1f})")
            else:
                fixed_p = fixed_prices[station]
                print(f"  Station {station} (COMPETITOR): ${fixed_p:.3f}/kWh (FIXED)")

    # Extract solution data - use aggregator stations for bounds info
    if verbose >= 1:
        print("Extracting solution data...")
    
    # Create bounds dictionaries for only aggregator stations for solution extraction
    agg_min_prices = {s: min_prices[s] for s in aggregator_stations}
    agg_max_prices = {s: max_prices[s] for s in aggregator_stations}
    
    solution_data = extract_aggregator_solution_data(final_model, aggregator_stations, agg_min_prices, agg_max_prices)
    
    if output_excel_file:
        if verbose >= 1:
            print(f"Saving solution to {output_excel_file}...")
        save_aggregator_solution_data(solution_data, output_excel_file)

    return {
        'solver_status': 'optimal',
        'objective_value': obj_value,
        'execution_time': results.solver.time if hasattr(results.solver, 'time') else None,
        'solution_data': solution_data,
        'charging_prices': charging_prices,
        'aggregator_stations': aggregator_stations,
        'competitor_stations': competitor_stations
    }
