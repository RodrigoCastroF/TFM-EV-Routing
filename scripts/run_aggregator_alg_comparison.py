"""
Comprehensive script to compare aggregator algorithms for different combinations of controlled stations.

This script:
1. Tests different combinations of controlled stations (e.g., 11,14,15), (11,14,37), (11,15), etc.
2. For each combination, runs the aggregator model with different algorithms
3. For each solution, tests it against the routing model to get real profit
4. Also tests base case scenario
5. Generates a CSV table with results comparing all algorithms
"""

from aggregator_model import solve_aggregator_model, load_aggregator_excel_data
from routing_model import load_excel_map_data, solve_for_all_evs, extract_electricity_costs
from utils import (
    TeeOutput, 
    get_price_info, 
    create_aggregator_data, 
    get_controlled_profit,
    solve_routing_and_get_profit,
    generate_station_combinations
)
import pandas as pd
import numpy as np
import os
import sys
from datetime import datetime


def run_experiment_for_combination(controlled_stations, base_case_prices, general_min_price, general_max_price,
                                 performance_csv_file, training_data_csv_file, base_map_file,
                                 all_stations, base_case_station_profits, algorithms, solver="gurobi", time_limit=300, verbose=1):
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
    
    if verbose >= 2:
        print(f"Price bounds: ${general_min_price:.3f} - ${general_max_price:.3f}")
        print(f"Controlled stations: {controlled_stations}")
        print(f"Competitor stations: {[s for s in base_case_prices.keys() if s not in controlled_stations]}")
    
    # Calculate base case profit from pre-computed results
    base_case_profit = get_controlled_profit(base_case_station_profits, controlled_stations)
    if verbose >= 1:
        print(f"Base case profit: ${base_case_profit:.4f}")

    # Store base case result
    controlled_stations_str = "|".join(map(str, controlled_stations))
    base_case_result = {
        'controlled_stations': controlled_stations_str,
        'type': 'base_case',
        'profit': base_case_profit
    }
    for station in all_stations:
        base_case_result[f'rc_{station}'] = base_case_prices[station]
    results.append(base_case_result)
    
    # Test each algorithm
    for alg in algorithms:
        if verbose >= 1:
            print()
            print(f"\n{'=' * 60}")
            print(f"Testing algorithm: {alg}")
            print(f"{'=' * 60}")
            print()
        
        # Solve aggregator model
        if verbose >= 2:
            print(f"\n{'=' * 40}")
            print(f"Solving aggregator model with {alg}...")
            print(f"{'=' * 40}")
        
        agg_results = solve_aggregator_model(
            input_data=synthetic_data,
            performance_csv_file=performance_csv_file,
            training_data_csv_file=training_data_csv_file,
            trust_region=False,  # No trust region for algorithm comparison
            alg=alg,
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
                    # Handle None prices for formatting
                    price_str = "None" if price is None else f"{price:.3f}"
                    solution_parts.append(f"{station}:{price_str}")
                solution_str = ", ".join(solution_parts)
                print(f"Solution prices: {solution_str}")
        
        # Store predicted result
        predicted_result = {
            'controlled_stations': controlled_stations_str,
            'type': f'{alg}_predicted',
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
        
        # Create a copy of solution_prices with None values replaced appropriately for routing model
        routing_prices = {}
        for station in all_stations:
            price = solution_prices[station]
            if price is None:
                if station in controlled_stations:
                    # For controlled stations, use minimum price if not set by optimizer
                    routing_prices[station] = general_min_price
                else:
                    # For competitor stations, use their fixed price from base case
                    routing_prices[station] = base_case_prices[station]
            else:
                routing_prices[station] = price
        
        real_profit = solve_routing_and_get_profit(routing_prices, controlled_stations, base_map_file, solver, time_limit, max(0, verbose-1))
        if verbose >= 1:
            print(f"Real profit: ${real_profit:.4f}")

        # Store real result
        real_result = {
            'controlled_stations': controlled_stations_str,
            'type': f'{alg}_real',
            'profit': real_profit
        }
        for station in all_stations:
            real_result[f'rc_{station}'] = solution_prices[station]
        results.append(real_result)
        
        if verbose >= 1:
            print(f"\n{'=' * 40}")
            print(f"RESULTS: {alg} algorithm")
            print(f"{'=' * 40}")
            print(f" Predicted profit: ${predicted_profit:.4f}")
            print(f" Real profit: ${real_profit:.4f}" if real_profit is not None else "    Real profit: N/A")
            if real_profit is not None and base_case_profit is not None:
                improvement = real_profit - base_case_profit
                if base_case_profit != 0:
                    percentage = improvement/base_case_profit*100
                    print(f" Improvement over base: ${improvement:.4f} ({percentage:.1f}%)")
                else:
                    print(f" Improvement over base: ${improvement:.4f} (base profit is zero)")
    
    return results


def main():
    """Main function to run all experiments."""
    # Configuration
    solver = "gurobi"
    time_limit = 15  # seconds
    verbose = 2  # 0=silent, 1=basic, 2=detailed
    
    # Available algorithms to test
    algorithms = ["linear", "rf", "svm", "cart", "gbm", "mlp"]
    
    # Input files
    base_aggregator_file = "../data/37-intersection map Aggregator Competition.xlsx"
    base_map_file = "../data/37-intersection map.xlsx"
    performance_csv_file = "../regressors/37map_1001scenarios_competition_performance_comparison.csv"
    training_data_csv_file = "../regressors/37map_1001scenarios_competition_training_data.csv"
    
    # Output files
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_csv_file = f"../results/aggregator_37map_alg_comparison_{timestamp}.csv"
    log_file_path = f"../logs/aggregator_37map_alg_comparison_{timestamp}.txt"
    
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
        print("AGGREGATOR ALGORITHM COMPARISON - STATION COMBINATIONS")
        print("=" * 80)
        print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Solver: {solver}")
        print(f"Time limit: {time_limit} seconds")
        print(f"Verbosity level: {verbose}")
        print(f"Algorithms to test: {', '.join(algorithms)}")
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
        combinations_to_test = generate_station_combinations(all_stations, min_size=1, max_size=5)
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
                algorithms=algorithms,
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
        print("ALGORITHM COMPARISON COMPLETED")
        print(f"{'='*80}")
        
    finally:
        # Restore original stdout
        if hasattr(sys.stdout, 'close'):
            sys.stdout.close()
        sys.stdout = original_stdout


if __name__ == "__main__":
    main() 