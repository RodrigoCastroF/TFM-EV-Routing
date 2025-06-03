from model import load_excel_map_data, solve_for_one_ev, solve_for_all_evs
import pandas as pd
import sys
import os
from datetime import datetime


class TeeOutput:
    """Class to write output to both console and file simultaneously."""

    def __init__(self, file_path):
        self.terminal = sys.stdout
        self.log_file = None
        if file_path is not None:
            self.log_file = open(file_path, 'w', encoding='utf-8')

    def write(self, message):
        self.terminal.write(message)
        if self.log_file:
            self.log_file.write(message)
            self.log_file.flush()  # Ensure immediate writing to file

    def flush(self):
        self.terminal.flush()
        if self.log_file:
            self.log_file.flush()

    def close(self):
        if self.log_file:
            self.log_file.close()


def main(input_excel_file, output_prefix=None, model_prefix=None, solver="gurobi", ev=None, scenario=None, scenarios_csv_file=None, time_limit=300, verbose=1, linearize_constraints=False, tuned_params_file=None, training_data=None):
    """
    Main function to solve EV routing problem.
    
    Args:
        input_excel_file: Path to input Excel file
        output_prefix: Prefix for output files (e.g., "../data/37-intersection map")
        model_prefix: Prefix for saving model in MPS format (optional, e.g., "../models/optimization")
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
        df = pd.read_csv(scenarios_csv_file)
        scenario_row = df[df['scenario'] == scenario]
        if scenario_row.empty:
            raise ValueError(f"Scenario {scenario} not found in {scenarios_csv_file}")
        
        # Extract charging prices (exclude the 'scenario' column)
        charging_stations = [col for col in df.columns if col != 'scenario']
        charging_prices = {}
        for station in charging_stations:
            charging_prices[int(station)] = float(scenario_row[station].iloc[0])
        if verbose >= 1:
            print(f"Charging prices for scenario {scenario}: {charging_prices}")
        
        # Update output prefix to include scenario
        if output_prefix:
            output_prefix = f"{output_prefix} S{scenario}"
    
    # Load data from Excel file once
    if verbose >= 1:
        print(f"Loading data from {input_excel_file}...")
    map_data = load_excel_map_data(input_excel_file, charging_prices=charging_prices, verbose=verbose)
    if verbose >= 1:
        print("Raw map data loaded successfully")
        print("List of EVs:", map_data["evs"])
        print("Charging prices:", map_data["charging_stations_df"]["pChargingPrice"].tolist())
    
    if ev is None:
        # Solve for all EVs using solve_for_all_evs
        if verbose >= 1:
            print("Solving for all EVs")
        
        results = solve_for_all_evs(
            map_data=map_data,
            output_prefix=output_prefix,
            model_prefix=model_prefix,
            solver=solver,
            time_limit=time_limit,
            verbose=verbose,
            linearize_constraints=linearize_constraints,
            tuned_params_file=tuned_params_file
        )
        
        # Save aggregated demand data to CSV if training_data is provided
        if training_data and "aggregated_demand" in results:
            aggregated_demand = results["aggregated_demand"]
            if aggregated_demand is not None:
                # Add scenario column
                scenario_value = 0 if scenario is None else scenario
                aggregated_demand['scenario'] = scenario_value
                
                # Convert charging_station to integer
                aggregated_demand['charging_station'] = aggregated_demand['charging_station'].astype(int)
                
                # Reorder columns to match required format
                columns_order = ['scenario', 'charging_station', 'time_period', 'aggregated_demand']
                aggregated_demand = aggregated_demand[columns_order]
                
                # Save to CSV
                if verbose >= 1:
                    print(f"\nSaving aggregated demand data to {training_data}...")
                try:
                    # Create directory if it doesn't exist
                    os.makedirs(os.path.dirname(training_data), exist_ok=True)
                    
                    # Check if file exists to determine if headers should be written
                    file_exists = os.path.isfile(training_data)
                    
                    # Write to CSV (append if file exists)
                    aggregated_demand.to_csv(training_data, mode='a' if file_exists else 'w', 
                                           index=False, header=not file_exists)
                    
                    if verbose >= 1:
                        print(f"Aggregated demand data saved successfully to {training_data}")
                except Exception as e:
                    if verbose >= 1:
                        print(f"Error saving aggregated demand data: {e}")
        
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
                print(f"EV Routing Solver Output - Scenario {scenario}, EV {current_ev} - {now}")
                print("=" * 60, "\n", sep="")

            # Generate output file paths if prefix provided
            output_excel_file = None
            output_image_file = None
            if output_prefix:
                output_excel_file = f"{output_prefix} EV{current_ev} Solution.xlsx"
                output_image_file = f"{output_prefix} EV{current_ev} Solution Map.png"
            
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
                tuned_params_file=tuned_params_file
            )
            results.append(result)

            if verbose >= 1:
                print("Final Results:", result)
                if log_file_path:
                    print(f"Output saved to: {log_file_path}")
                print("\n" + "=" * 60)
        
        # Return single result for single EV, list of results for multiple EVs
        return results[0] if isinstance(ev, int) else results


if __name__ == "__main__":

    # Configuration
    linearize_constraints = True
    solver = "gurobi"
    scenarios = [0]
    # evs = [1]
    evs = None  # Solve for all EVs
    time_limit = 15

    # Input files
    input_excel_file = "../data/37-intersection map.xlsx"
    scenarios_csv_file = "../data/scenarios.csv"

    # Output files
    output_prefix = f"../data/37-intersection map{' LIN' if linearize_constraints else ''}{' CPLEX' if solver == 'cplex' else ''}"
    # output_prefix = None  # Avoid saving files
    training_data_path = "../data/training_data.csv"

    # Detailed logging
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file_path = f"../data/routing_solver_output_{timestamp}.txt"
    # log_file_path = None
    
    # Set up output redirection to both console and file
    if log_file_path:
        os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
    tee_output = TeeOutput(log_file_path)
    original_stdout = sys.stdout
    sys.stdout = tee_output
    
    try:
        for scenario in scenarios:
            main(
                scenario=scenario,
                ev=evs,
                output_prefix=output_prefix,
                linearize_constraints=linearize_constraints,
                solver=solver,
                time_limit=time_limit,
                verbose=2,
                input_excel_file=input_excel_file,
                scenarios_csv_file=scenarios_csv_file,
                training_data=training_data_path,
                # model_prefix=output_prefix,
                # tuned_params_file="../data/tuned_params_1.prm"
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

