from routing_model import load_excel_map_data, solve_for_one_ev, solve_for_all_evs, extract_electricity_costs
import pandas as pd
import sys
import os
import traceback
from datetime import datetime
from utils import TeeOutput


def main(input_excel_file, output_prefix_solution=None, output_prefix_image=None, model_prefix=None, solver="gurobi", ev=None, scenario=None, scenarios_csv_file=None, time_limit=300, verbose=1, linearize_constraints=False, tuned_params_file=None, training_data=None, load_if_exists=False):
    """
    Main function to solve EV routing problem.
    
    Args:
        input_excel_file: Path to input Excel file
        output_prefix_solution: Prefix for solution files (e.g., "../data/37-intersection map")
        output_prefix_image: Prefix for image files (e.g., "../data/37-intersection map")
        model_prefix: Prefix for saving routing_model in MPS format (optional, e.g., "../models/optimization")
        solver: Solver to use (default: "gurobi")
        ev: Specific EV(s) to solve for. Can be:
            - None: solve for all EVs using solve_for_all_evs
            - int: solve for a single EV using solve_for_one_ev
            - list of int: solve for each EV in the list using solve_for_one_ev
        scenario: Scenario number to use for charging prices (requires scenarios_csv_file)
        scenarios_csv_file: Path to CSV file containing scenarios with charging prices
        time_limit: Time limit in seconds (default: 300)
        verbose: Verbosity level (0=silent, 1=basic, 2=detailed)
        linearize_constraints: Whether to use linearized constraints (default: False)
        tuned_params_file: Path to tuned parameters file (.prm) for Gurobi (optional)
        training_data: Path to save aggregated demand data as CSV (optional)
        load_if_exists: Whether to load existing solutions from Excel files if they exist (default: False)
    
    Returns:
        Dictionary with results (single result for one EV, list of results for multiple EVs)
    """
    
    # Handle scenario-based charging prices
    charging_prices = None
    if scenario is not None:
        if scenarios_csv_file is None:
            raise ValueError("scenarios_csv_file must be provided when scenario is specified")
        if verbose >= 1:
            print(f"Loading charging prices for scenario {scenario} from {scenarios_csv_file}...")
        
        # Load scenario charging prices
        scenarios_df = pd.read_csv(scenarios_csv_file)
        scenario_row = scenarios_df[scenarios_df['scenario'] == scenario]
        if scenario_row.empty:
            raise ValueError(f"Scenario {scenario} not found in {scenarios_csv_file}")
        
        # Extract charging prices (exclude the 'scenario' column)
        charging_stations = [col for col in scenarios_df.columns if col != 'scenario']
        charging_prices = {}
        for station in charging_stations:
            charging_prices[int(station)] = float(scenario_row[station].iloc[0])
        if verbose >= 1:
            print(f"Charging prices for scenario {scenario}: {charging_prices}")
        
        # Update output prefixes to include scenario
        if output_prefix_solution:
            output_prefix_solution = f"{output_prefix_solution} S{scenario}"
        if output_prefix_image:
            output_prefix_image = f"{output_prefix_image} S{scenario}"
    
    # Load data from Excel file once
    if verbose >= 1:
        print(f"Loading data from {input_excel_file}...")
    map_data = load_excel_map_data(input_excel_file, charging_prices=charging_prices, verbose=verbose)
    electricity_costs = extract_electricity_costs(map_data)
    if verbose >= 1:
        print("Raw map data loaded successfully")
        print("List of EVs:", map_data["evs"])
        print("Charging prices:", map_data["charging_stations_df"]["pChargingPrice"].tolist())
        print("Electricity costs:", electricity_costs)
    
    if ev is None:
        # Solve for all EVs using solve_for_all_evs
        if verbose >= 1:
            print("\n\n", "=" * 100, sep="")
            now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            constraints_str = "linear" if linearize_constraints else "quadratic"
            print(f"EV Routing Solver Output - Solver {solver}, Constraints {constraints_str}, Scenario {scenario}, All EVs - {now}")
            print("=" * 100, "\n", sep="")
        
        results = solve_for_all_evs(
            map_data=map_data,
            output_prefix_solution=output_prefix_solution,
            output_prefix_image=output_prefix_image,
            model_prefix=model_prefix,
            solver=solver,
            time_limit=time_limit,
            verbose=verbose,
            linearize_constraints=linearize_constraints,
            tuned_params_file=tuned_params_file,
            load_if_exists=load_if_exists
        )
        
        # Process aggregated demand data if available
        if "aggregated_demand" in results and results["aggregated_demand"] is not None:
            aggregated_demand = results["aggregated_demand"].copy()
            scenario_value = 0 if scenario is None else scenario
            aggregated_demand['scenario'] = scenario_value
            aggregated_demand['charging_station'] = aggregated_demand['charging_station'].astype(int)
            columns_order = ['scenario', 'charging_station', 'time_period', 'aggregated_demand']
            processed_demand_df = aggregated_demand[columns_order]
        else:
            if verbose >= 1:
                print("Warning: Aggregated demand not available in results.")
            processed_demand_df = None
        
        # Save aggregated demand data to CSV if training_data is provided
        if training_data and processed_demand_df is not None:
            if verbose >= 1:
                print(f"\nChecking for existing data and saving aggregated demand data to {training_data}...")
            try:
                # Create directory if it doesn't exist
                os.makedirs(os.path.dirname(training_data), exist_ok=True)
                
                # Check if file exists and if scenario data already exists
                file_exists = os.path.isfile(training_data)
                scenario_exists = False
                
                if file_exists:
                    # Read existing data to check for duplicates
                    try:
                        existing_df = pd.read_csv(training_data)
                        if 'scenario' in existing_df.columns:
                            scenario_value = 0 if scenario is None else scenario
                            scenario_exists = scenario_value in existing_df['scenario'].values
                            if scenario_exists:
                                if verbose >= 1:
                                    print(f"Scenario {scenario_value} already exists in training data. Skipping duplicate entry.")
                    except Exception as read_error:
                        if verbose >= 1:
                            print(f"Warning: Could not read existing file to check for duplicates: {read_error}")
                        # Continue with normal append behavior if we can't read the file
                
                # Only write if scenario doesn't already exist
                if not scenario_exists:
                    # Write to CSV (append if file exists)
                    processed_demand_df.to_csv(training_data, mode='a' if file_exists else 'w', 
                                           index=False, header=not file_exists)
                    if verbose >= 1:
                        scenario_value = 0 if scenario is None else scenario
                        print(f"Aggregated demand data for scenario {scenario_value} saved successfully to {training_data}")
                
            except Exception as e:
                if verbose >= 1:
                    print(f"Error saving aggregated demand data: {e}")
        
        # Display profit results if computed
        if "station_profits" in results and results["station_profits"] is not None:
            station_profits = results["station_profits"]
            total_profit = sum(station_profits.values())
            if verbose >= 1:
                print(f"\nProfit Results:")
                for station, profit in station_profits.items():
                    print(f"  Station {station}: ${profit:.4f}")
                print(f"Total Scenario Profit: ${total_profit:.4f}")

            # Add total profit to results for compatibility
            results['scenario_profit'] = total_profit
        else:
            if verbose >= 1:
                print("Warning: Station profits not available in results.")
            results['scenario_profit'] = None
        
        return results
    else:
        # Convert single EV to list for uniform handling
        ev_list = [ev] if isinstance(ev, int) else ev
        
        # Validate all EVs exist in data
        invalid_evs = [e for e in ev_list if e not in map_data["evs"]]
        if invalid_evs:
            raise ValueError(f"EVs {invalid_evs} not found in data. Available EVs: {map_data['evs']}")
        
        if verbose >= 1:
            print(f"Solving for EV(s): {ev_list}")
        
        # Solve for each EV in the list
        results = []
        for current_ev in ev_list:

            if verbose >= 1:
                print("\n\n", "=" * 60, sep="")
                now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                constraints_str = "linear" if linearize_constraints else "quadratic"
                print(f"EV Routing Solver Output - Solver {solver}, Constraints {constraints_str}, Scenario {scenario}, EV {current_ev} - {now}")
                print("=" * 60, "\n", sep="")

            # Generate output file paths if prefixes provided
            output_excel_file = None
            output_image_file = None
            if output_prefix_solution:
                output_excel_file = f"{output_prefix_solution} EV{current_ev} Solution.xlsx"
            if output_prefix_image:
                output_image_file = f"{output_prefix_image} EV{current_ev} Solution Map.png"
            
            result = solve_for_one_ev(
                map_data=map_data,
                ev=current_ev,
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
            results.append(result)

            if verbose >= 1:
                print("Final Results:", result)
                print("\n" + "=" * 60)
        
        # Return single result for single EV, list of results for multiple EVs
        return results[0] if isinstance(ev, int) else results


if __name__ == "__main__":

    # Configuration
    solvers = ["gurobi"]  # solvers = ["gurobi", "cplex"]
    linearization = [True]  # linearization = [True, False]
    scenarios = [1]
    evs = None  # evs = None to solve for all EVs; evs = [1] to solve for a specific EV
    time_limit = 15
    load_if_exists = True

    # Input files
    input_excel_file = "../data/37-intersection map.xlsx"
    scenarios_csv_file = "../data/scenarios.csv"

    # Detailed logging
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file_path = f"../logs/routing_solver_output_{timestamp}.txt"
    # log_file_path = None

    # Set up output redirection to both console and file
    if log_file_path:
        os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
    tee_output = TeeOutput(log_file_path)
    original_stdout = sys.stdout
    sys.stdout = tee_output

    try:
        for solver in solvers:
            linearization_options = linearization if solver == "gurobi" else [True]
            for linearize_constraints in linearization_options:

                # Output files
                sol_name = f"37-intersection map{' LIN' if linearize_constraints else ''}{' CPLEX' if solver == 'cplex' else ''}"
                output_prefix_solution = f"../solutions/{sol_name}"
                # output_prefix_solution = None  # Avoid saving solution
                output_prefix_image = f"../images/{sol_name}"
                # output_prefix_image = None  # Avoid saving image
                # output_prefix_model = f"../gurobi_parameters/{sol_name}"
                output_prefix_model = None  # Avoid saving concrete model
                # training_data_path = "../data/training_data.csv"
                training_data_path = None  # Avoid updating training data

                for scenario in scenarios:
                    main(
                        scenario=scenario,
                        ev=evs,
                        output_prefix_solution=output_prefix_solution,
                        output_prefix_image=output_prefix_image,
                        linearize_constraints=linearize_constraints,
                        load_if_exists=load_if_exists,
                        solver=solver,
                        time_limit=time_limit,
                        verbose=2,
                        input_excel_file=input_excel_file,
                        scenarios_csv_file=scenarios_csv_file,
                        training_data=training_data_path,
                        # model_prefix=output_prefix_model,
                        # tuned_params_file="../gurobi_parameters/tuned_params_1.prm"
                    )

    except Exception as e:
        print(f"Error during execution: {e}")
        import traceback
        traceback.print_exc()

    finally:
        # Restore original stdout and close the log file
        sys.stdout = original_stdout
        tee_output.close()
        if log_file_path:
            print(f"All output has been saved to: {log_file_path}")

