from model import load_excel_map_data, filter_map_data_for_ev, get_ev_routing_abstract_model, extract_solution_data, save_solution_data, create_solution_map
import pyomo.environ as pyo
from pyomo.opt import SolverFactory


def solve_for_one_ev(input_data, ev, output_excel_file=None, output_image_file=None, model_prefix=None, solver="gurobi", time_limit=300, verbose=1, linearize_constraints=False, tuned_params_file=None):
    """
    Solve the EV routing problem for a single EV.
    
    Args:
        input_data: Filtered data for the specific EV
        ev: EV number
        output_excel_file: Path to save Excel solution (optional)
        output_image_file: Path to save solution map image (optional)
        model_prefix: Prefix for saving model in MPS format (optional, e.g., "../models/optimization")
        solver: Solver to use (default: "gurobi")
        time_limit: Time limit in seconds (default: 300)
        verbose: Verbosity level (0=silent, 1=basic, 2=detailed)
        linearize_constraints: Whether to use linearized constraints (default: False)
        tuned_params_file: Path to tuned parameters file (.prm) for Gurobi (optional)
    
    Returns:
        Dictionary with solution results
    """
    
    # Get the abstract model
    if verbose >= 1:
        constraint_type = "linearized" if linearize_constraints else "quadratic"
        print(f"Creating abstract model for EV {ev} with {constraint_type} constraints...")
    abstract_model = get_ev_routing_abstract_model(linearize_constraints=linearize_constraints)
    
    # Create a concrete instance using the data
    if verbose >= 1:
        print(f"Creating concrete model instance for EV {ev}...")
    concrete_model = abstract_model.create_instance(input_data)
    
    # Save model in MPS format if requested
    if model_prefix:
        model_file = f"{model_prefix} EV{ev} Model.mps"
        if verbose >= 1:
            print(f"Saving concrete model for EV {ev} to {model_file}...")
        try:
            concrete_model.write(model_file)
            if verbose >= 1:
                print(f"Model for EV {ev} saved successfully in MPS format!")
        except Exception as e:
            if verbose >= 1:
                print(f"Error saving model for EV {ev}: {e}")
    
    # Basic model information
    if verbose >= 1:
        print(f"\nModel Information for EV {ev}:")
        print(f"Number of intersections: {len(concrete_model.sIntersections)}")
        print(f"Number of paths: {len(concrete_model.sPaths)}")
        print(f"Number of delivery points: {len(concrete_model.sDeliveryPoints)}")
        print(f"Number of charging stations: {len(concrete_model.sChargingStations)}")
    
    # Create solver instance
    if verbose >= 1:
        tuned_msg = f" with tuned parameters from {tuned_params_file}" if tuned_params_file else ""
        print(f"\nSetting up {solver} solver for EV {ev}{tuned_msg}...")
    opt = SolverFactory(solver)
    
    # Set time limit based on solver
    time_limit_option = {"cbc": "seconds", "gurobi": "timeLimit", "glpk": "tmlim"}
    if solver in time_limit_option:
        opt.options[time_limit_option[solver]] = time_limit
        if verbose >= 2:
            print(f"Time limit set to {time_limit} seconds")
    
    # Load tuned parameters for Gurobi if provided
    if tuned_params_file and solver == "gurobi":
        import os
        if os.path.exists(tuned_params_file):
            if verbose >= 1:
                print(f"Loading tuned parameters from {tuned_params_file}...")
            try:
                # Read the parameter file and apply to solver options
                with open(tuned_params_file, 'r') as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith('#'):
                            parts = line.split()
                            if len(parts) >= 2:
                                param = parts[0]
                                value = parts[1]
                                
                                # Convert value to appropriate type
                                try:
                                    if '.' in value:
                                        value = float(value)
                                    else:
                                        value = int(value)
                                except ValueError:
                                    pass  # Keep as string
                                
                                opt.options[param] = value
                                if verbose >= 2:
                                    print(f"  Set {param} = {value}")
                
                if verbose >= 1:
                    print("Tuned parameters loaded successfully!")
            except Exception as e:
                if verbose >= 1:
                    print(f"Warning: Could not load tuned parameters: {e}")
        else:
            if verbose >= 1:
                print(f"Warning: Tuned parameters file not found: {tuned_params_file}")
    
    # Solve the model
    if verbose >= 1:
        print(f"Solving the model for EV {ev}...")
    results = opt.solve(concrete_model, tee=(verbose >= 2))
    
    if verbose >= 2:
        print(f"\nSOLVER RESULTS for EV {ev}:")
        print(results)

    # Handle the case where no solution object exists
    if results.solver.status != pyo.SolverStatus.ok and results.solver.status != pyo.SolverStatus.aborted:
        if verbose >= 1:
            print(f"\nSolver returned no solution for EV {ev} :(")
            print(f"\tStatus: {results.solver.status}")
            print(f"\tTermination condition: {results.solver.termination_condition}")
        return {'ev': ev, 'solver_status': 'no_solution'}

    # At this point, a solution object should exist
    if verbose >= 1:
        print(f"\nSolver returned a solution for EV {ev}! :)")
        print(f"\tStatus: {results.solver.status}")
        print(f"\tTermination condition: {results.solver.termination_condition}")

    # Extract solution information
    try:
        lower_bound = results['problem'][0]['Lower bound']
        upper_bound = results['problem'][0]['Upper bound']
        if lower_bound is not None and lower_bound != 0:
            final_gap = (upper_bound - lower_bound) / abs(upper_bound)
        else:
            final_gap = 0.0
    except (KeyError, TypeError, IndexError):
        final_gap = 0.0
        lower_bound = None
        upper_bound = None

    try:
        execution_time = results['solver'][0]['Time']
    except (KeyError, IndexError, AttributeError):
        execution_time = None

    # Get objective function value
    obj_value = pyo.value(concrete_model.Obj)

    if verbose >= 1:
        print(f"Objective function value for EV {ev}: {obj_value}")
        if final_gap is not None and verbose >= 2:
            print(f"Final gap: {final_gap:2.1%}")
        if execution_time is not None and verbose >= 2:
            print(f"Execution time: {execution_time:.2f} seconds")

    if verbose >= 1:
        print(f"\nExtracting solution data for EV {ev}...")
    solution_data = extract_solution_data(concrete_model)
    if verbose >= 1:
        print(f"Solution data extracted successfully for EV {ev}!")

    # Save solution data to Excel if file path provided
    if output_excel_file:
        if verbose >= 1:
            print(f"\nSaving solution data for EV {ev} to {output_excel_file}...")
        try:
            save_solution_data(solution_data, output_excel_file)
            if verbose >= 1:
                print(f"Solution data for EV {ev} saved successfully!")
        except Exception as e:
            if verbose >= 1:
                print(f"Error saving solution data for EV {ev}: {e}")

    # Create solution map visualization if file path provided
    if output_image_file:
        if verbose >= 1:
            print(f"\nCreating solution map visualization for EV {ev}: {output_image_file}...")
        try:
            create_solution_map(solution_data, input_data, output_image_file, ev=ev)
            if verbose >= 1:
                print(f"Solution map for EV {ev} created successfully!")
        except Exception as e:
            if verbose >= 1:
                print(f"Error creating solution map for EV {ev}: {e}")

    # Return results summary
    ev_results = {
        'ev': ev,
        'solution_data': solution_data,
        'solver_status': results.solver.status,
        'termination_condition': results.solver.termination_condition,
        'objective_value': obj_value,
        'final_gap': final_gap,
        'execution_time': execution_time,
        'lower_bound': lower_bound,
        'upper_bound': upper_bound
    }

    return ev_results


def solve_for_all_evs(input_excel_file, output_prefix=None, model_prefix=None, solver="gurobi", time_limit=300, verbose=1, linearize_constraints=False, tuned_params_file=None):
    """
    Solve the EV routing problem for all EVs in the dataset.
    
    Args:
        input_excel_file: Path to input Excel file
        output_prefix: Prefix for output files (e.g., "../data/37-intersection map")
        model_prefix: Prefix for saving models in MPS format (optional, e.g., "../models/optimization")
        solver: Solver to use (default: "gurobi")
        time_limit: Time limit in seconds per EV (default: 300)
        verbose: Verbosity level (0=silent, 1=basic, 2=detailed)
        linearize_constraints: Whether to use linearized constraints (default: False)
        tuned_params_file: Path to tuned parameters file (.prm) for Gurobi (optional)
    
    Returns:
        Dictionary with results for all EVs
    """
    
    # Load data from Excel file
    if verbose >= 1:
        print(f"Loading data from {input_excel_file}...")
    map_data = load_excel_map_data(input_excel_file)
    if verbose >= 1:
        print("Raw map data loaded successfully")
        print("List of EVs:", map_data["evs"])
    
    all_results = {}
    
    # Solve for each EV
    for ev in map_data["evs"]:
        if verbose >= 1:
            print(f"\n{'='*50}")
            print(f"Processing EV {ev}")
            print(f"{'='*50}")
        
        # Filter data for specific EV
        if verbose >= 1:
            print(f"Filtering data for EV {ev}...")
        input_data = filter_map_data_for_ev(map_data, ev)
        if verbose >= 2:
            print("Input data filtered successfully")
        
        # Generate output file paths if prefix provided
        output_excel_file = None
        output_image_file = None
        if output_prefix:
            output_excel_file = f"{output_prefix} EV{ev} Solution.xlsx"
            output_image_file = f"{output_prefix} EV{ev} Solution Map.png"
        
        # Solve for this EV
        ev_results = solve_for_one_ev(
            input_data=input_data,
            ev=ev,
            output_excel_file=output_excel_file,
            output_image_file=output_image_file,
            model_prefix=model_prefix,
            solver=solver,
            time_limit=time_limit,
            verbose=verbose,
            linearize_constraints=linearize_constraints,
            tuned_params_file=tuned_params_file
        )
        
        all_results[ev] = ev_results
    
    if verbose >= 1:
        print(f"\n{'='*50}")
        print("SUMMARY OF ALL EVs")
        print(f"{'='*50}")
        for ev, results in all_results.items():
            if 'objective_value' in results:
                print(f"EV {ev}: Objective = {results['objective_value']:.2f}")
            else:
                print(f"EV {ev}: {results.get('solver_status', 'unknown status')}")
    
    return all_results


def main(input_excel_file, output_prefix=None, model_prefix=None, solver="gurobi", ev=None, time_limit=300, verbose=1, linearize_constraints=False, tuned_params_file=None):
    """
    Main function to solve EV routing problem.
    
    Args:
        input_excel_file: Path to input Excel file
        output_prefix: Prefix for output files (e.g., "../data/37-intersection map")
        model_prefix: Prefix for saving model in MPS format (optional, e.g., "../models/optimization")
        solver: Solver to use (default: "gurobi")
        ev: Specific EV to solve for (if None, solve for all EVs)
        time_limit: Time limit in seconds (default: 300)
        verbose: Verbosity level (0=silent, 1=basic, 2=detailed)
        linearize_constraints: Whether to use linearized constraints (default: False)
        tuned_params_file: Path to tuned parameters file (.prm) for Gurobi (optional)
    
    Returns:
        Dictionary with results
    """
    
    if ev is not None:
        # Solve for specific EV
        if verbose >= 1:
            print(f"Solving for specific EV: {ev}")
        
        # Load and filter data
        map_data = load_excel_map_data(input_excel_file)
        if ev not in map_data["evs"]:
            print(f"Error: EV {ev} not found in data. Available EVs: {map_data['evs']}")
            return None
        
        input_data = filter_map_data_for_ev(map_data, ev)
        
        # Generate output file paths if prefix provided
        output_excel_file = None
        output_image_file = None
        if output_prefix:
            output_excel_file = f"{output_prefix} EV{ev} Solution.xlsx"
            output_image_file = f"{output_prefix} EV{ev} Solution Map.png"
        
        return solve_for_one_ev(
            input_data=input_data,
            ev=ev,
            output_excel_file=output_excel_file,
            output_image_file=output_image_file,
            model_prefix=model_prefix,
            solver=solver,
            time_limit=time_limit,
            verbose=verbose,
            linearize_constraints=linearize_constraints,
            tuned_params_file=tuned_params_file
        )
    else:
        # Solve for all EVs
        if verbose >= 1:
            print("Solving for all EVs")
        
        return solve_for_all_evs(
            input_excel_file=input_excel_file,
            output_prefix=output_prefix,
            model_prefix=model_prefix,
            solver=solver,
            time_limit=time_limit,
            verbose=verbose,
            linearize_constraints=linearize_constraints,
            tuned_params_file=tuned_params_file
        )


if __name__ == "__main__":
    linearize_constraints = True
    input_excel_file = "../data/37-intersection map.xlsx"
    output_prefix = "../data/37-intersection map LIN" if linearize_constraints else "../data/37-intersection map"
    
    # Solve for all EVs
    results = main(
        input_excel_file=input_excel_file,
        output_prefix=output_prefix,
        time_limit=15,
        verbose=2,
        linearize_constraints=linearize_constraints,
        ev=1,
        model_prefix=output_prefix,
        # tuned_params_file="../data/tuned_params_1.prm"
    )
    print("Final Results:", results)

