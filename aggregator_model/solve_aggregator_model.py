"""
This function solves the simplified aggregator model with embedded OptiCL regression
"""

from .get_aggregator_map_data import load_aggregator_excel_data
from .save_aggregator_solution_data import extract_aggregator_solution_data, save_aggregator_solution_data
import pyomo.environ as pyo
from pyomo.opt import SolverFactory
import pandas as pd
import opticl


def solve_aggregator_model(input_excel_file, performance_csv_file, training_data_csv_file, trust_region=True,
                          output_excel_file=None, solver="gurobi", time_limit=300, verbose=1):
    """
    Solve the simplified aggregator optimization problem with embedded regression model.

    Parameters
    ----------
    input_excel_file: str
        Path to the input Excel file containing aggregator data (charging station bounds).
    performance_csv_file: str
        Path to the CSV file containing performance comparison of regression models.
    training_data_csv_file: str
        Path to the CSV file containing training data for the regression models.
    trust_region: bool
        If True, solutions are restricted to lie in the trust region (the domain of the training data)
    output_excel_file: str, optional
        Path to save the solution Excel file (optional).
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

    # Load charging station bounds from Excel file
    if verbose >= 1:
        print(f"Loading aggregator data from {input_excel_file}...")
    input_data = load_aggregator_excel_data(input_excel_file, verbose=verbose)
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

    # Select best regression model
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
