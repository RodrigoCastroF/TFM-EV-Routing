import pandas as pd
import numpy as np
import os


def generate_scenarios(output_csv_file, num_scenarios=1000, seed=42):
    """
    Generate scenarios with charging prices for each charging station.
    If the CSV file already exists, new scenarios will be appended with incremented IDs.
    
    Args:
        output_csv_file: Path to save the scenarios CSV file
        num_scenarios: Number of scenarios to generate (default: 1000)
        seed: Random seed for reproducibility (default: 42)
    """
    
    # Set random seed for reproducibility
    np.random.seed(seed)
    
    # Charging station IDs
    charging_stations = [11, 14, 15, 26, 37]
    
    # Check if CSV file already exists
    existing_df = None
    start_scenario_id = 0
    
    if os.path.exists(output_csv_file):
        try:
            existing_df = pd.read_csv(output_csv_file)
            if not existing_df.empty:
                # Ensure existing DataFrame has the correct columns
                # Convert charging station IDs to strings since CSV stores them as strings
                expected_columns = ['scenario'] + [str(station) for station in charging_stations]
                if list(existing_df.columns) != expected_columns:
                    print(f"Warning: Existing CSV has different columns. Expected: {expected_columns}")
                    print(f"Found: {list(existing_df.columns)}")
                    # Try to select only the expected columns if they exist
                    if all(col in existing_df.columns for col in expected_columns):
                        existing_df = existing_df[expected_columns]
                        print("Selected matching columns from existing data")
                    else:
                        print("Cannot match columns. Will create new file.")
                        existing_df = None
                        start_scenario_id = 0
                
                if existing_df is not None:
                    # Get the highest existing scenario ID
                    max_existing_id = existing_df['scenario'].max()
                    start_scenario_id = max_existing_id + 1
                    print(f"Found existing scenarios (0-{max_existing_id}). New scenarios will start from ID {start_scenario_id}")
            else:
                print("Found empty CSV file. Starting from scenario 0")
        except Exception as e:
            print(f"Warning: Could not read existing CSV file: {e}")
            print("Will create new file starting from scenario 0")
            existing_df = None
            start_scenario_id = 0
    else:
        print("CSV file doesn't exist. Creating new file starting from scenario 0")
    
    # Initialize scenarios list
    scenarios = []
    
    # Only add baseline scenario if we're starting from 0 (no existing scenarios)
    if start_scenario_id == 0:
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
        scenario_range = range(1, num_scenarios)
    else:
        # Generate scenarios starting from start_scenario_id
        scenario_range = range(start_scenario_id, start_scenario_id + num_scenarios)
    
    for scenario in scenario_range:
        # Generate random prices for each charging station
        # Uniform distribution between 0.2 and 0.8 $/kWh
        prices = np.random.uniform(0.2, 0.8, len(charging_stations))
        scenario_row = [scenario] + prices.tolist()
        scenarios.append(scenario_row)
    
    # Create DataFrame for new scenarios
    # Use string column names to match CSV format
    columns = ['scenario'] + [str(station) for station in charging_stations]
    new_df = pd.DataFrame(scenarios, columns=columns)
    
    # Round prices to 3 decimal places for cleaner output
    for station in charging_stations:
        new_df[str(station)] = new_df[str(station)].round(3)
    
    # Combine with existing data if it exists
    if existing_df is not None and not existing_df.empty:
        # Ensure both DataFrames have the same column order
        existing_df = existing_df[columns]
        new_df = new_df[columns]
        df = pd.concat([existing_df, new_df], ignore_index=True)
        total_scenarios = len(df)
        print(f"Added {len(new_df)} new scenarios to existing {len(existing_df)} scenarios")
    else:
        df = new_df
        total_scenarios = len(df)
        print(f"Created {total_scenarios} new scenarios")
    
    # Save to CSV
    df.to_csv(output_csv_file, index=False)
    
    print(f"Total scenarios in file: {total_scenarios}")
    print(f"Saved to {output_csv_file}")
    
    if start_scenario_id == 0:
        baseline_prices = {
            11: 0.6,
            14: 0.6,
            15: 0.5,
            26: 0.6,
            37: 0.5
        }
        print(f"Baseline scenario (0): {dict(zip(charging_stations, [baseline_prices[station] for station in charging_stations]))}")
    
    print(f"Price range for random scenarios: 0.2 - 0.8 $/kWh")
    
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
    # Filter out any non-numeric column names that might be duplicates
    charging_stations = []
    for col in df.columns:
        if col != 'scenario':
            try:
                # Try to convert column name to int to ensure it's a valid station ID
                station_id = int(float(col))  # Use float first to handle cases like "11.0"
                charging_stations.append(station_id)
            except (ValueError, TypeError):
                print(f"Warning: Skipping invalid column name: {col}")
                continue
    
    charging_prices = {}
    
    for station in charging_stations:
        # Use string representation of station ID to access the column
        col_name = str(station)
        if col_name in df.columns:
            charging_prices[station] = float(scenario_row[col_name].iloc[0])
        else:
            # Try alternative column name formats
            for possible_col in df.columns:
                try:
                    if int(float(possible_col)) == station:
                        charging_prices[station] = float(scenario_row[possible_col].iloc[0])
                        break
                except (ValueError, TypeError):
                    continue
    
    return charging_prices


if __name__ == "__main__":
    # Generate scenarios and save to CSV
    output_file = "../data/scenarios.csv"
    scenarios_df = generate_scenarios(output_file, num_scenarios=9000, seed=42)
    
    # Display first few scenarios as example
    print("\nFirst 5 scenarios:")
    print(scenarios_df.head())
    
    # Test loading a specific scenario
    print(f"\nTesting scenario loading...")
    scenario_0_prices = load_scenario_charging_prices(output_file, 0)
    print(f"Scenario 0 charging prices: {scenario_0_prices}")
    
    scenario_1_prices = load_scenario_charging_prices(output_file, 1)
    print(f"Scenario 1 charging prices: {scenario_1_prices}") 