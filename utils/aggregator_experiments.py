"""
Shared utilities for aggregator experiments.

This module contains functions that are common to both:
- run_aggregator_experiments.py
- run_aggregator_alg_comparison.py
"""

import pandas as pd
import numpy as np
from itertools import combinations
from routing_model import load_excel_map_data, solve_for_all_evs


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


def generate_station_combinations(all_stations, min_size, max_size):
    """Generate all combinations of stations to test."""
    combinations_to_test = []
    for size in range(min_size, max_size + 1):
        for combo in combinations(all_stations, size):
            combinations_to_test.append(list(combo))
    return combinations_to_test 