from .get_routing_map_data import filter_map_data_for_ev
from .get_routing_abstract_model import get_ev_routing_abstract_model
from .save_ev_solution_data import extract_solution_data, save_solution_data, create_solution_map, load_solution_data
from .save_scenario_solution_data import extract_aggregated_demand, create_scenario_analysis_plots
import pyomo.environ as pyo
from pyomo.opt import SolverFactory
import os


def solve_for_one_ev(map_data, ev, output_excel_file=None, output_image_file=None, model_prefix=None, solver="gurobi",
                     time_limit=300, verbose=1, linearize_constraints=False, tuned_params_file=None, load_if_exists=False):
    """
    Solve the EV routing problem for a single EV.

    Args:
        map_data: Raw map data object returned by load_excel_map_data
        ev: EV number
        output_excel_file: Path to save Excel solution (optional)
        output_image_file: Path to save solution map image (optional)
        model_prefix: Prefix for saving routing_model in MPS format (optional, e.g., "../models/optimization")
        solver: Solver to use (default: "gurobi")
        time_limit: Time limit in seconds (default: 300)
        verbose: Verbosity level (0=silent, 1=basic, 2=detailed)
        linearize_constraints: Whether to use linearized constraints (default: False)
        tuned_params_file: Path to tuned parameters file (.prm) for Gurobi (optional)
        load_if_exists: Whether to load existing solution from Excel file if it exists (default: False)

    Returns:
        Dictionary with solution results
    """

    # Check if we should load existing solution
    if load_if_exists and output_excel_file and os.path.exists(output_excel_file):
        if verbose >= 1:
            print(f"Loading existing solution for EV {ev} from {output_excel_file}...")
        try:
            solution_data, metadata = load_solution_data(output_excel_file)
            
            # Reconstruct results with loaded data
            ev_results = {'ev': ev, 'solution_data': solution_data}
            if metadata:
                ev_results.update(metadata)
            else:
                # Set default values if metadata is missing
                ev_results.update({
                    'solver_status': None,
                    'termination_condition': None,
                    'objective_value': None,
                    'final_gap': None,
                    'execution_time': None,
                    'lower_bound': None,
                    'upper_bound': None
                })
            
            if verbose >= 1:
                print(f"Solution for EV {ev} loaded successfully!")
                if 'objective_value' in ev_results and ev_results['objective_value'] is not None:
                    print(f"Objective function value for EV {ev}: {ev_results['objective_value']}")
            
            # Create solution map if requested
            if output_image_file:
                input_data = filter_map_data_for_ev(map_data, ev)
                if verbose >= 1:
                    print(f"Creating solution map visualization for EV {ev}: {output_image_file}...")
                try:
                    create_solution_map(solution_data, input_data, output_image_file, ev=ev)
                    if verbose >= 1:
                        print(f"Solution map for EV {ev} created successfully!")
                except Exception as e:
                    if verbose >= 1:
                        print(f"Error creating solution map for EV {ev}: {e}")
            
            return ev_results
            
        except Exception as e:
            if verbose >= 1:
                print(f"Error loading solution for EV {ev}: {e}")
                print(f"Falling back to solving the model...")

    # Filter data for the specific EV
    if verbose >= 1:
        print(f"Filtering data for EV {ev}...")
    input_data = filter_map_data_for_ev(map_data, ev)
    if verbose >= 2:
        print("Input data filtered successfully")

    # Get the abstract routing_model
    if verbose >= 1:
        constraint_type = "linearized" if linearize_constraints else "quadratic"
        print(f"Creating abstract routing_model for EV {ev} with {constraint_type} constraints...")
    abstract_model = get_ev_routing_abstract_model(linearize_constraints=linearize_constraints)

    # Create a concrete instance using the data
    if verbose >= 1:
        print(f"Creating concrete routing_model instance for EV {ev}...")
    concrete_model = abstract_model.create_instance(input_data)

    # Save routing_model in MPS format if requested
    if model_prefix:
        model_file = f"{model_prefix} EV{ev} Model.mps"
        if verbose >= 1:
            print(f"Saving concrete routing_model for EV {ev} to {model_file}...")
        try:
            concrete_model.write(model_file)
            if verbose >= 1:
                print(f"Model for EV {ev} saved successfully in MPS format!")
        except Exception as e:
            if verbose >= 1:
                print(f"Error saving routing_model for EV {ev}: {e}")

    # Basic routing_model information
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
    time_limit_option = {"cbc": "seconds", "gurobi": "timeLimit", "glpk": "tmlim", "cplex": "timelimit"}
    if solver in time_limit_option:
        opt.options[time_limit_option[solver]] = time_limit
        if verbose >= 2:
            print(f"Time limit set to {time_limit} seconds")

    # Load tuned parameters for Gurobi if provided
    if tuned_params_file and solver == "gurobi":
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

    # Solve the routing_model
    if verbose >= 1:
        print(f"Solving the routing_model for EV {ev}...")
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
            # Prepare metadata for saving
            metadata = {
                'solver_status': str(results.solver.status),
                'termination_condition': str(results.solver.termination_condition),
                'objective_value': obj_value,
                'final_gap': final_gap,
                'execution_time': execution_time,
                'lower_bound': lower_bound,
                'upper_bound': upper_bound
            }
            save_solution_data(solution_data, output_excel_file, metadata=metadata)
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


def solve_for_all_evs(map_data, output_prefix_solution=None, output_prefix_image=None, model_prefix=None, solver="gurobi", time_limit=300, verbose=1,
                      linearize_constraints=False, tuned_params_file=None, load_if_exists=False):
    """
    Solve the EV routing problem for all EVs in the dataset.

    Args:
        map_data: Raw map data object returned by load_excel_map_data
        output_prefix_solution: Prefix for solution files (e.g., "../data/37-intersection map")
        output_prefix_image: Prefix for image files (e.g., "../data/37-intersection map")
        model_prefix: Prefix for saving models in MPS format (optional, e.g., "../models/optimization")
        solver: Solver to use (default: "gurobi")
        time_limit: Time limit in seconds per EV (default: 300)
        verbose: Verbosity level (0=silent, 1=basic, 2=detailed)
        linearize_constraints: Whether to use linearized constraints (default: False)
        tuned_params_file: Path to tuned parameters file (.prm) for Gurobi (optional)
        load_if_exists: Whether to load existing solutions from Excel files if they exist (default: False)

    Returns:
        Dictionary with results for all EVs
    """

    all_results = {}

    # Solve for each EV
    for ev in map_data["evs"]:
        if verbose >= 1:
            print(f"\n{'=' * 50}")
            print(f"Processing EV {ev}")
            print(f"{'=' * 50}")

        # Generate output file paths if prefixes provided
        output_excel_file = None
        output_image_file = None
        if output_prefix_solution:
            output_excel_file = f"{output_prefix_solution} EV{ev} Solution.xlsx"
        if output_prefix_image:
            output_image_file = f"{output_prefix_image} EV{ev} Solution Map.png"

        # Solve for this EV
        ev_results = solve_for_one_ev(
            map_data=map_data,
            ev=ev,
            output_excel_file=output_excel_file,
            output_image_file=output_image_file,
            model_prefix=model_prefix,
            solver=solver,
            time_limit=time_limit,
            verbose=verbose,
            linearize_constraints=linearize_constraints,
            tuned_params_file=tuned_params_file,
            load_if_exists=load_if_exists
        )

        all_results[ev] = ev_results

    if verbose >= 1:
        print(f"\n{'=' * 50}")
        print("SUMMARY OF ALL EVs")
        print(f"{'=' * 50}")
        for ev, results in all_results.items():
            if 'objective_value' in results and results['objective_value'] is not None:
                print(f"EV {ev}: Objective = {results['objective_value']:.2f}")
            else:
                print(f"EV {ev}: {results.get('solver_status', 'unknown status')}")

    if verbose >= 1:
        print(f"\n{'=' * 50}")
        print("Extracting aggregated demand...")
        print(f"{'=' * 50}")
    
    # Extract aggregated demand for all EVs
    try:
        aggregated_demand = extract_aggregated_demand(all_results, map_data, verbose=verbose)
        if verbose >= 1:
            print("Aggregated demand extracted successfully!")
    except Exception as e:
        if verbose >= 1:
            print(f"Error extracting aggregated demand: {e}")
        aggregated_demand = None
    all_results["aggregated_demand"] = aggregated_demand

    # Create scenario analysis plots if output prefix is provided
    if output_prefix_image:
        if verbose >= 1:
            print(f"\n{'=' * 50}")
            print("Creating scenario analysis plots...")
            print(f"{'=' * 50}")
        try:
            output_plot_file = f"{output_prefix_image} Scenario Analysis.png"
            create_scenario_analysis_plots(all_results, map_data, output_plot_file, aggregated_demand=aggregated_demand, verbose=verbose)
            if verbose >= 1:
                print("Scenario analysis plots created successfully!")
        except Exception as e:
            if verbose >= 1:
                print(f"Error creating scenario analysis plots: {e}")

    return all_results