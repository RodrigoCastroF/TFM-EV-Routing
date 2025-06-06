"""
This function solves the aggregator model by combining data loading, model creation, solving, and solution saving
"""

from .get_aggregator_map_data import load_aggregator_excel_data
from .get_aggregator_abstract_model import get_aggregator_abstract_model
from .save_aggregator_solution_data import extract_aggregator_solution_data, save_aggregator_solution_data
import pyomo.environ as pyo
from pyomo.opt import SolverFactory


def solve_aggregator_model(input_excel_file, output_excel_file=None, solver="gurobi", time_limit=300, verbose=1):
    """
    Solve the aggregator optimization problem.

    Parameters
    ----------
    input_excel_file: str
        Path to the input Excel file containing aggregator data.
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

    # Load data from Excel file
    if verbose >= 1:
        print(f"Loading aggregator data from {input_excel_file}...")
    input_data = load_aggregator_excel_data(input_excel_file, verbose=verbose)
    if verbose >= 1:
        print("Aggregator data loaded successfully")

    # Get the abstract model
    if verbose >= 1:
        print("Creating abstract aggregator model...")
    abstract_model = get_aggregator_abstract_model()

    # Create a concrete instance using the data
    if verbose >= 1:
        print("Creating concrete model instance...")
    concrete_model = abstract_model.create_instance(input_data)

    # Basic model information
    if verbose >= 1:
        print(f"\nModel Information:")
        print(f"Number of charging stations: {len(concrete_model.sChargingStations)}")
        print(f"Number of time periods: {len(concrete_model.sTimePeriods)}")
        print(f"Charging stations: {list(concrete_model.sChargingStations)}")
        print(f"Time periods: {list(concrete_model.sTimePeriods)}")

    # Create solver instance
    if verbose >= 1:
        print(f"\nSetting up {solver} solver...")
    opt = SolverFactory(solver)

    # Set time limit based on solver
    time_limit_option = {"cbc": "seconds", "gurobi": "timeLimit", "glpk": "tmlim", "cplex": "timelimit"}
    if solver in time_limit_option:
        opt.options[time_limit_option[solver]] = time_limit
        if verbose >= 2:
            print(f"Time limit set to {time_limit} seconds")

    # Solve the model
    if verbose >= 1:
        print("Solving the aggregator model...")
    results = opt.solve(concrete_model, tee=(verbose >= 2))

    if verbose >= 2:
        print(f"\nSOLVER RESULTS:")
        print(results)

    # Handle the case where no solution object exists
    if results.solver.status != pyo.SolverStatus.ok and results.solver.status != pyo.SolverStatus.aborted:
        if verbose >= 1:
            print(f"\nSolver returned no solution :(")
            print(f"\tStatus: {results.solver.status}")
            print(f"\tTermination condition: {results.solver.termination_condition}")
        return {'solver_status': 'no_solution'}

    # At this point, a solution object should exist
    if verbose >= 1:
        print(f"\nSolver returned a solution! :)")
        print(f"\tStatus: {results.solver.status}")
        print(f"\tTermination condition: {results.solver.termination_condition}")

    # Extract solution information
    try:
        execution_time = results['solver'][0]['Time']
    except (KeyError, IndexError, AttributeError):
        execution_time = None

    # Get objective function value
    obj_value = pyo.value(concrete_model.Obj)

    if verbose >= 1:
        print(f"Objective function value (total profit): ${obj_value:.2f}")
        if execution_time is not None and verbose >= 2:
            print(f"Execution time: {execution_time:.2f} seconds")

    # Display optimal charging prices
    if verbose >= 1:
        print(f"\nOptimal charging prices:")
        for station in concrete_model.sChargingStations:
            price = pyo.value(concrete_model.vChargingPrice[station])
            min_price = concrete_model.pMinChargingPrice[station]
            max_price = concrete_model.pMaxChargingPrice[station]
            print(f"  Station {station}: ${price:.3f}/kWh (range: ${min_price:.1f}-${max_price:.1f})")

    if verbose >= 1:
        print("Extracting solution data...")
    solution_data = extract_aggregator_solution_data(concrete_model)
    if verbose >= 1:
        print("Solution data extracted successfully!")

    # Save solution data to Excel if file path provided
    if output_excel_file:
        if verbose >= 1:
            print(f"\nSaving solution data to {output_excel_file}...")
        try:
            save_aggregator_solution_data(solution_data, output_excel_file)
            if verbose >= 1:
                print("Solution data saved successfully!")
        except Exception as e:
            if verbose >= 1:
                print(f"Error saving solution data: {e}")

    # Return results
    return {
        'solver_status': 'optimal',
        'objective_value': obj_value,
        'execution_time': execution_time,
        'solution_data': solution_data,
        'charging_prices': {station: pyo.value(concrete_model.vChargingPrice[station]) 
                           for station in concrete_model.sChargingStations}
    }
