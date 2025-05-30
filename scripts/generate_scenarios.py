import pandas as pd
import numpy as np


def generate_scenarios(output_csv_file, num_scenarios=1000, seed=42):
    """
    Generate scenarios with charging prices for each charging station.
    
    Args:
        output_csv_file: Path to save the scenarios CSV file
        num_scenarios: Number of scenarios to generate (default: 1000)
        seed: Random seed for reproducibility (default: 42)
    """
    
    # Set random seed for reproducibility
    np.random.seed(seed)
    
    # Charging station IDs
    charging_stations = [11, 14, 15, 26, 37]
    
    # Initialize scenarios list
    scenarios = []
    
    # Scenario 0: Baseline scenario
    baseline_prices = {
        11: 0.6,
        14: 0.6,
        15: 0.5,
        26: 0.6,
        37: 0.5
    }
    baseline_row = [0] + [baseline_prices[station] for station in charging_stations]
    scenarios.append(baseline_row)
    
    # Generate remaining scenarios (1 to num_scenarios-1)
    for scenario in range(1, num_scenarios):
        # Generate random prices for each charging station
        # Uniform distribution between 0.2 and 0.8 $/kWh
        prices = np.random.uniform(0.2, 0.8, len(charging_stations))
        scenario_row = [scenario] + prices.tolist()
        scenarios.append(scenario_row)
    
    # Create DataFrame
    columns = ['scenario'] + charging_stations
    df = pd.DataFrame(scenarios, columns=columns)
    
    # Round prices to 3 decimal places for cleaner output
    for station in charging_stations:
        df[station] = df[station].round(3)
    
    # Save to CSV
    df.to_csv(output_csv_file, index=False)
    
    print(f"Generated {num_scenarios} scenarios and saved to {output_csv_file}")
    print(f"Baseline scenario (0): {dict(zip(charging_stations, [baseline_prices[station] for station in charging_stations]))}")
    print(f"Price range for scenarios 1-{num_scenarios-1}: 0.2 - 0.8 $/kWh")
    
    return df


def load_scenario_charging_prices(scenarios_csv_file, scenario):
    """
    Load charging prices for a specific scenario from the CSV file.
    
    Args:
        scenarios_csv_file: Path to the scenarios CSV file
        scenario: Scenario number to load
        
    Returns:
        Dictionary with {charging_station: charging_price} format
    """
    
    # Load scenarios DataFrame
    df = pd.read_csv(scenarios_csv_file)
    
    # Find the row for the specified scenario
    scenario_row = df[df['scenario'] == scenario]
    
    if scenario_row.empty:
        raise ValueError(f"Scenario {scenario} not found in {scenarios_csv_file}")
    
    # Extract charging prices (exclude the 'scenario' column)
    charging_stations = [col for col in df.columns if col != 'scenario']
    charging_prices = {}
    
    for station in charging_stations:
        charging_prices[int(station)] = float(scenario_row[station].iloc[0])
    
    return charging_prices


if __name__ == "__main__":
    # Generate scenarios and save to CSV
    output_file = "../data/scenarios.csv"
    scenarios_df = generate_scenarios(output_file, num_scenarios=1000, seed=42)
    
    # Display first few scenarios as example
    print("\nFirst 5 scenarios:")
    print(scenarios_df.head())
    
    # Test loading a specific scenario
    print(f"\nTesting scenario loading...")
    scenario_0_prices = load_scenario_charging_prices(output_file, 0)
    print(f"Scenario 0 charging prices: {scenario_0_prices}")
    
    scenario_1_prices = load_scenario_charging_prices(output_file, 1)
    print(f"Scenario 1 charging prices: {scenario_1_prices}") 