"""
Comprehensive script to run aggregator experiments for different combinations of controlled stations.

This script:
1. Tests different combinations of controlled stations (e.g., 11,14,15), (11,14,37), (11,15), etc.
2. For each combination, runs the aggregator model with and without trust region
3. For each solution, tests it against the routing model to get real profit
4. Also tests base case and max prices scenarios
5. Generates a CSV table with results
"""

from aggregator_model import solve_aggregator_model, load_aggregator_excel_data
from routing_model import load_excel_map_data, solve_for_all_evs, extract_electricity_costs
from utils import TeeOutput
import pandas as pd
import numpy as np
import os
import sys
from datetime import datetime
from itertools import combinations


def get_price_info(base_aggregator_data, base_map_data):
    """Extract all price information needed for experiments."""
    # Base case prices from map data
    base_case_prices = {}
    for _, row in base_map_data['charging_stations_df'].iterrows():
        station_id = int(row['pStationIntersection'])
        base_case_prices[station_id] = row['pChargingPrice']
    
    # General min/max prices from aggregator data
    all_base_stations = base_aggregator_data[None]['sChargingStations'][None]
    min_prices_dict = base_aggregator_data[None]['pMinChargingPrice']
    max_prices_dict = base_aggregator_data[None]['pMaxChargingPrice']
    
    # Extract general values (should be 0.2 and 0.8)
    general_min_price = next(min_prices_dict[s] for s in all_base_stations if s in min_prices_dict)
    general_max_price = next(max_prices_dict[s] for s in all_base_stations if s in max_prices_dict)
    
    return base_case_prices, general_min_price, general_max_price


def create_aggregator_data(controlled_stations, base_case_prices, general_min_price, general_max_price):
    """Create synthetic aggregator instance data."""
    all_stations = list(base_case_prices.keys())
    
    synthetic_data = {None: {
        'sChargingStations': {None: all_stations},
        'pMinChargingPrice': {},
        'pMaxChargingPrice': {},
        'pChargingPrice': {}
    }}
    
    for station in all_stations:
        if station in controlled_stations:
            # Controlled stations: set min/max bounds, nan for fixed price
            synthetic_data[None]['pMinChargingPrice'][station] = general_min_price
            synthetic_data[None]['pMaxChargingPrice'][station] = general_max_price
            synthetic_data[None]['pChargingPrice'][station] = np.nan
        else:
            # Competitor stations: set fixed prices, nan for min/max bounds
            synthetic_data[None]['pMinChargingPrice'][station] = np.nan
            synthetic_data[None]['pMaxChargingPrice'][station] = np.nan
            synthetic_data[None]['pChargingPrice'][station] = base_case_prices[station]
    
    return synthetic_data


def get_controlled_profit(station_profits, controlled_stations):
    """Sum profits of controlled stations only."""
    return sum(station_profits.get(str(station), 0) for station in controlled_stations)


def solve_routing_and_get_profit(charging_prices, controlled_stations, base_map_file, solver, time_limit, verbose):
    """Solve routing model with given prices and return profit of controlled stations."""
    if verbose >= 1:
        print(f"\nSolving routing model for prices {charging_prices}...")
        print()
    
    map_data = load_excel_map_data(base_map_file, charging_prices=charging_prices, verbose=0)
    results = solve_for_all_evs(
        map_data,
        solver=solver,
        time_limit=time_limit,
        verbose=verbose,
        linearize_constraints=True
    )
    
    if "station_profits" in results and results["station_profits"] is not None:
        profit = get_controlled_profit(results["station_profits"], controlled_stations)
        if verbose >= 2:
            print(f"→ Routing profit: ${profit:.4f}")
        return profit
    
    if verbose >= 2:
        print(f"→ Routing failed - no profit data")
    return None


def run_experiment_for_combination(controlled_stations, base_case_prices, general_min_price, general_max_price,
                                 performance_csv_file, training_data_csv_file, base_map_file,
                                 all_stations, base_case_station_profits, solver="gurobi", time_limit=300, verbose=1):
    """Run complete experiment for a specific combination of controlled stations."""
    results = []
    
    if verbose >= 1:
        print()
        print(f"\n{'='*60}")
        print(f"EXPERIMENT: Controlled stations {controlled_stations}")
        print(f"{'='*60}")
        print()
    
    # Create synthetic aggregator instance
    if verbose >= 2:
        print(f"Creating synthetic aggregator data for stations {controlled_stations}...")
    synthetic_data = create_aggregator_data(controlled_stations, base_case_prices, general_min_price, general_max_price)
    
    # Create max prices scenario
    max_case_prices = base_case_prices.copy()
    for station in controlled_stations:
        max_case_prices[station] = general_max_price
    
    if verbose >= 2:
        print(f"Price bounds: ${general_min_price:.3f} - ${general_max_price:.3f}")
        print(f"Controlled stations: {controlled_stations}")
        print(f"Competitor stations: {[s for s in base_case_prices.keys() if s not in controlled_stations]}")
    
    # Calculate base case profit from pre-computed results
    base_case_profit = get_controlled_profit(base_case_station_profits, controlled_stations)
    if verbose >= 1:
        print(f"Base case profit: ${base_case_profit:.4f}")

    # Run max prices once per combination  
    if verbose >= 1:
        print(f"\n{'=' * 40}")
        print(f"Testing max prices scenario...")
        print(f"{'=' * 40}")
    max_prices_profit = solve_routing_and_get_profit(max_case_prices, controlled_stations, base_map_file, solver, time_limit, max(0, verbose-1))
    if verbose >= 1:
        print(f"Max prices profit: ${max_prices_profit:.4f}")

    # Store base case and max prices results
    controlled_stations_str = "|".join(map(str, controlled_stations))
    
    # Base case result
    base_case_result = {
        'controlled_stations': controlled_stations_str,
        'type': 'base_case',
        'profit': base_case_profit
    }
    for station in all_stations:
        base_case_result[f'rc_{station}'] = base_case_prices[station]
    results.append(base_case_result)
    
    # Max prices result
    max_prices_result = {
        'controlled_stations': controlled_stations_str,
        'type': 'max_prices', 
        'profit': max_prices_profit
    }
    for station in all_stations:
        max_prices_result[f'rc_{station}'] = max_case_prices[station]
    results.append(max_prices_result)
    
    # Test both trust region settings
    for trust_region in [True, False]:
        tr_str = "with" if trust_region else "without"
        type_prefix = "sol_tr" if trust_region else "sol"
        
        if verbose >= 1:
            print()
            print(f"\n{'=' * 60}")
            print(f"Testing {tr_str} trust region")
            print(f"{'=' * 60}")
            print()
        
        # Solve aggregator model
        if verbose >= 2:
            print(f"\n{'=' * 40}")
            print(f"Solving aggregator model...")
            print(f"{'=' * 40}")
        
        agg_results = solve_aggregator_model(
            input_data=synthetic_data,
            performance_csv_file=performance_csv_file,
            training_data_csv_file=training_data_csv_file,
            trust_region=trust_region,
            model="competition",
            solver=solver,
            time_limit=time_limit,
            verbose=max(0, verbose-1)
        )
        
        predicted_profit = agg_results['objective_value']
        solution_prices = agg_results['charging_prices']
        
        if verbose >= 2:
            print(f"Aggregator solver status: {agg_results.get('solver_status', 'unknown')}")
        
        if verbose >= 1:
            print(f"Predicted profit: ${predicted_profit:.2f}")
            if verbose >= 2:
                solution_parts = []
                for station in sorted(all_stations):
                    price = solution_prices[station]
                    solution_parts.append(f"{station}:{price:.3f}")
                solution_str = ", ".join(solution_parts)
                print(f"Solution prices: {solution_str}")
        
        # Store predicted result
        predicted_result = {
            'controlled_stations': controlled_stations_str,
            'type': f'{type_prefix}_predicted',
            'profit': predicted_profit
        }
        for station in all_stations:
            predicted_result[f'rc_{station}'] = solution_prices[station]
        results.append(predicted_result)
        
        # Test solution against routing model
        if verbose >= 1:
            print(f"\n{'=' * 40}")
            print(f"Testing solution against routing model...")
            print(f"{'=' * 40}")
        real_profit = solve_routing_and_get_profit(solution_prices, controlled_stations, base_map_file, solver, time_limit, max(0, verbose-1))
        if verbose >= 1:
            print(f"Real profit: ${real_profit:.4f}")

        # Store real result
        real_result = {
            'controlled_stations': controlled_stations_str,
            'type': f'{type_prefix}_real',
            'profit': real_profit
        }
        for station in all_stations:
            real_result[f'rc_{station}'] = solution_prices[station]
        results.append(real_result)
        
        if verbose >= 1:
            print(f"\n{'=' * 40}")
            print(f"RESULTS: {tr_str} trust_region")
            print(f"{'=' * 40}")
            print(f" Predicted profit: ${predicted_profit:.4f}")
            print(f" Real profit: ${real_profit:.4f}" if real_profit is not None else "    Real profit: N/A")
            if real_profit is not None and base_case_profit is not None:
                improvement = real_profit - base_case_profit
                print(f" Improvement over base: ${improvement:.4f} ({improvement/base_case_profit*100:.1f}%)")
    
    return results


def generate_station_combinations(all_stations, min_size, max_size):
    """Generate all combinations of stations to test."""
    combinations_to_test = []
    for size in range(min_size, max_size + 1):
        for combo in combinations(all_stations, size):
            combinations_to_test.append(list(combo))
    return combinations_to_test


def main():
    """Main function to run all experiments."""
    # Configuration
    solver = "gurobi"
    time_limit = 15  # seconds
    verbose = 2  # 0=silent, 1=basic, 2=detailed
    
    # Input files
    base_aggregator_file = "../data/37-intersection map Aggregator Competition.xlsx"
    base_map_file = "../data/37-intersection map.xlsx"
    performance_csv_file = "../regressors/37map_1001scenarios_competition_performance_comparison.csv"
    training_data_csv_file = "../regressors/37map_1001scenarios_competition_training_data.csv"
    
    # Output files
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_csv_file = f"../results/aggregator_experiments_{timestamp}.csv"
    log_file_path = f"../logs/aggregator_experiments_{timestamp}.txt"
    
    # Create output directories if they don't exist
    os.makedirs(os.path.dirname(output_csv_file), exist_ok=True)
    os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
    
    # Set up output redirection to save to log file
    original_stdout = sys.stdout
    sys.stdout = TeeOutput(log_file_path)
    
    try:
        # Check if input files exist
        for file_path in [base_aggregator_file, base_map_file, performance_csv_file, training_data_csv_file]:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"Input file not found: {file_path}")
        
        print("=" * 80)
        print("AGGREGATOR EXPERIMENTS - STATION COMBINATIONS")
        print("=" * 80)
        print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Solver: {solver}")
        print(f"Time limit: {time_limit} seconds")
        print(f"Verbosity level: {verbose}")
        print(f"Output CSV: {output_csv_file}")
        print(f"Log file: {log_file_path}")
        print()
        
        # Load base data and extract price information
        print("Loading base data...")
        print(f"→ Loading aggregator data from: {base_aggregator_file}")
        base_aggregator_data = load_aggregator_excel_data(base_aggregator_file, verbose=0)
        print(f"→ Loading map data from: {base_map_file}")
        base_map_data = load_excel_map_data(base_map_file, verbose=0)
        print(f"→ Extracting price information...")
        base_case_prices, general_min_price, general_max_price = get_price_info(base_aggregator_data, base_map_data)
        
        # Get all available stations for combinations
        all_stations = base_aggregator_data[None]['sChargingStations'][None]
        print(f"→ Available stations: {all_stations}")
        print(f"→ Base case prices: {base_case_prices}")
        print(f"→ Price range: ${general_min_price:.3f} - ${general_max_price:.3f}")
        print()
        
        # Generate combinations to test
        print("Generating station combinations...")
        combinations_to_test = generate_station_combinations(all_stations, min_size=4, max_size=5)
        print(f"→ Testing {len(combinations_to_test)} combinations:")
        for i, combo in enumerate(combinations_to_test, 1):
            print(f"    {i:2d}. {combo}")
        print()
        
        # Run base case once for all combinations
        print("Running base case scenario (once for all combinations)...")
        print("=" * 80)
        map_data = load_excel_map_data(base_map_file, charging_prices=base_case_prices, verbose=0)
        base_case_results = solve_for_all_evs(
            map_data,
            solver=solver,
            time_limit=time_limit,
            verbose=max(0, verbose-1),
            linearize_constraints=True
        )
        
        if "station_profits" not in base_case_results or base_case_results["station_profits"] is None:
            raise RuntimeError("Failed to solve base case - no station profits available")
        
        base_case_station_profits = base_case_results["station_profits"]
        print(f"✓ Base case solved successfully")
        print()

        # Run experiments
        print("Starting experiments...")
        print("=" * 80)
        all_results = []
        
        for i, controlled_stations in enumerate(combinations_to_test):
            print(f"\nPROGRESS: Combination {i+1}/{len(combinations_to_test)} - {controlled_stations}")
            print(f"Remaining: {len(combinations_to_test) - i - 1} combinations")
            
            combo_results = run_experiment_for_combination(
                controlled_stations=controlled_stations,
                base_case_prices=base_case_prices,
                general_min_price=general_min_price,
                general_max_price=general_max_price,
                performance_csv_file=performance_csv_file,
                training_data_csv_file=training_data_csv_file,
                base_map_file=base_map_file,
                all_stations=all_stations,
                base_case_station_profits=base_case_station_profits,
                solver=solver,
                time_limit=time_limit,
                verbose=verbose
            )
            
            all_results.extend(combo_results)
            print(f"✓ Completed combination {i+1}/{len(combinations_to_test)}")
        
        # Create results DataFrame and save to CSV
        if all_results:
            print(f"\n{'='*80}")
            print("SAVING RESULTS")
            print(f"{'='*80}")
            
            results_df = pd.DataFrame(all_results)
            results_df.to_csv(output_csv_file, index=False)
            
            print(f"Total experiments completed: {len(all_results)}")
            print(f"Results saved to: {output_csv_file}")
            print(f"Log saved to: {log_file_path}")
            print()
            print("Results preview:")
            print("-" * 80)
            print(results_df.to_string(index=False))
            print("-" * 80)
        else:
            print(f"\nERROR: No results generated!")
        
        print(f"\n{'='*80}")
        print("EXPERIMENTS COMPLETED")
        print(f"{'='*80}")
        
    finally:
        # Restore original stdout
        if hasattr(sys.stdout, 'close'):
            sys.stdout.close()
        sys.stdout = original_stdout


if __name__ == "__main__":
    main()
