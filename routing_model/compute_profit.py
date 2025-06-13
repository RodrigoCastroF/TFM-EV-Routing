"""
This file provides utility functions to compute the profit of charging stations for a routing model solution
"""


def compute_profit(charging_prices, aggregated_demand, electricity_costs, verbose=0):
    """
    Compute profit for each charging station given charging prices, aggregated demand, and electricity costs.

    Parameters:
    -----------
    charging_prices : dict
        Dictionary mapping charging station to charging price {station_id: price}
    aggregated_demand : pd.DataFrame
        DataFrame containing aggregated demand data with columns:
        ['charging_station', 'time_period', 'aggregated_demand']
    electricity_costs : dict
        Dictionary mapping time_period to electricity cost (C_t)
    verbose : int, default=0
        Verbosity level (0=silent, 1=detailed computations)

    Returns:
    --------
    dict
        Dictionary mapping station to profit: {station_id: profit}
    """
    if verbose >= 1:
        print(f"\n--- Profit Computation ---")
        print(f"Charging prices: {charging_prices}")
        print(f"Total demand records: {len(aggregated_demand)}")

    # Initialize station profits dictionary
    station_profits = {}
    station_revenues = {}
    station_costs = {}
    
    # For verbose output
    revenue_details = {} if verbose >= 1 else None
    cost_details = {} if verbose >= 1 else None
    total_revenue = 0 if verbose >= 1 else None
    total_cost = 0 if verbose >= 1 else None

    # Calculate revenue and cost for each station
    for _, row in aggregated_demand.iterrows():
        station = str(int(row['charging_station']))
        demand = row['aggregated_demand']
        time_period = int(row['time_period'])
        
        # Skip if station not in charging prices (shouldn't happen in normal usage)
        if station not in charging_prices:
            continue
        
        # Initialize station data if not exists
        if station not in station_profits:
            station_profits[station] = 0
            station_revenues[station] = 0
            station_costs[station] = 0
        
        # Calculate revenue for this station
        price = charging_prices[station]
        station_revenue = price * demand
        station_revenues[station] += station_revenue
        
        # Calculate cost for this station
        elec_cost = electricity_costs[time_period]
        station_cost = elec_cost * demand
        station_costs[station] += station_cost
        
        # Calculate profit for this station
        station_profits[station] = station_revenues[station] - station_costs[station]
        
        # Verbose output collection
        if verbose >= 1:
            total_revenue += station_revenue
            total_cost += station_cost
            
            if demand > 0:  # Only show non-zero demand
                # Revenue details
                if station not in revenue_details:
                    revenue_details[station] = []
                revenue_details[station].append(
                    f"t{time_period}:{demand:.3f}kWh*${price:.3f}=${station_revenue:.4f}")
                
                # Cost details
                key = f"t{time_period}"
                if key not in cost_details:
                    cost_details[key] = []
                cost_details[key].append(f"s{station}:{demand:.3f}kWh*${elec_cost:.3f}=${station_cost:.4f}")

    if verbose >= 1:
        print(f"\nRevenue breakdown by station:")
        for station, details in revenue_details.items():
            print(f"  Station {station}: {' + '.join(details)}")
        print(f"\nCost breakdown by time period:")
        for period, details in cost_details.items():
            print(f"  {period}: {' + '.join(details)}")
        print(f"\nProfit breakdown by station:")
        for station, profit in station_profits.items():
            print(f"  Station {station}: ${profit:.4f}")
        print(f"\nTotal Revenue: ${total_revenue:.4f}")
        print(f"Total Cost: ${total_cost:.4f}")
        print(f"Profit: ${sum(station_profits.values()):.4f}")
        print(f"--- End Profit Computation ---\n")

    return station_profits


def compute_profit_stations(scenario, demand_df, scenarios_df, electricity_costs, verbose=0):
    """
    Compute profit for each charging station in a scenario.

    Parameters:
    -----------
    scenario : int
        Scenario ID to compute profit for
    demand_df : pd.DataFrame
        DataFrame containing aggregated demand data with columns:
        ['scenario', 'charging_station', 'time_period', 'aggregated_demand']
    scenarios_df : pd.DataFrame
        DataFrame containing charging prices for each scenario
    electricity_costs : dict
        Dictionary mapping time_period to electricity cost (C_t)
    verbose : int, default=0
        Verbosity level (0=silent, 1=detailed computations)

    Returns:
    --------
    dict
        Dictionary mapping station to profit: {station_id: profit}
    """
    # Get charging stations from scenarios dataframe
    charging_stations = [col for col in scenarios_df.columns if col != 'scenario']

    # Check if scenario exists in both dataframes
    if scenario not in scenarios_df['scenario'].values:
        raise ValueError(f"Scenario {scenario} not found in scenarios dataframe")

    if scenario not in demand_df['scenario'].values:
        raise ValueError(f"Scenario {scenario} not found in demand dataframe")

    # Get charging prices for this scenario
    scenario_prices = scenarios_df[scenarios_df['scenario'] == scenario]
    prices = {station: scenario_prices[str(station)].iloc[0] for station in charging_stations}

    # Get demand data for this scenario
    scenario_demand = demand_df[demand_df['scenario'] == scenario]

    if verbose >= 1:
        print(f"\n--- Scenario {scenario} Profit Computation ---")

    # Use the new compute_profit function
    station_profits = compute_profit(prices, scenario_demand, electricity_costs, verbose)
    
    if verbose >= 1:
        print(f"--- End Scenario {scenario} Profit Computation ---\n")

    return station_profits


def compute_scenario_profit(scenario, demand_df, scenarios_df, electricity_costs, verbose=0):
    """
    Compute profit for a single scenario.

    Parameters:
    -----------
    scenario : int
        Scenario ID to compute profit for
    demand_df : pd.DataFrame
        DataFrame containing aggregated demand data with columns:
        ['scenario', 'charging_station', 'time_period', 'aggregated_demand']
    scenarios_df : pd.DataFrame
        DataFrame containing charging prices for each scenario
    electricity_costs : dict
        Dictionary mapping time_period to electricity cost (C_t)
    verbose : int, default=0
        Verbosity level (0=silent, 1=detailed computations)

    Returns:
    --------
    float
        Profit for the scenario (revenue - cost)
    """
    # Use the new function to get station profits (with verbose output handled there)
    station_profits = compute_profit_stations(scenario, demand_df, scenarios_df, electricity_costs, verbose)
    
    # Calculate total profit by summing all station profits
    return sum(station_profits.values())
