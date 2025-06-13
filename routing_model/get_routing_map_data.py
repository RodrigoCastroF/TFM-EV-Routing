"""
These functions handle the input data of the routing routing_model,
loading the map data of a scenario and filtering it for a single EV
"""


import pandas as pd


def load_excel_map_data(file_path: str, charging_prices: dict = None, verbose: int = 0) -> dict:
    """
    Load data from an Excel file for the EV routing optimization routing_model.

    Parameters
    ----------
    file_path: str
        The path to the Excel file (.xlsx) containing the data.
    charging_prices: dict, optional
        Dictionary with format {charging_station: charging_price} to override 
        the pChargingPrice values from the Excel file. Default is None.
    verbose: int
        Verbosity level.

    Returns
    -------
    map_data: dict
        The raw map data loaded from Excel, containing all dataframes, metadata, list of EVs, and coordinates.
    """

    # Read sheets from the Excel file
    unindexed_df = pd.read_excel(file_path, sheet_name="Unindexed", index_col="Name")
    paths_df = pd.read_excel(file_path, sheet_name="sPaths")
    delivery_points_df = pd.read_excel(file_path, sheet_name="sDeliveryPoints")
    charging_stations_df = pd.read_excel(file_path, sheet_name="sChargingStations")
    time_periods_df = pd.read_excel(file_path, sheet_name="sTimePeriods")
    
    # Try to read coordinates if the sheet exists
    coordinates = None
    try:
        coordinates_df = pd.read_excel(file_path, sheet_name="Coordinates")
        # Convert to dictionary format: {node_id: (x, y)}
        coordinates = {}
        for _, row in coordinates_df.iterrows():
            coordinates[int(row['Node'])] = (row['X'], row['Y'])
    except Exception as e:
        print(f"Warning: Could not read coordinates from Excel file: {e}")
        print("Coordinates will not be available for visualization")

    # Function to clean column names by taking only the first word
    def clean_column_name(col_name):
        return col_name.split(" ")[0]

    # Clean column names for all dataframes
    for df in [paths_df, delivery_points_df, charging_stations_df, time_periods_df]:
        df.columns = [clean_column_name(col) for col in df.columns]
    
    # Clean the index names in unindexed_df
    unindexed_df.index = [clean_column_name(idx) for idx in unindexed_df.index]

    # Override charging prices if provided
    if charging_prices is not None:
        # Create a mapping from charging station intersection to row index
        station_to_index = {}
        for idx, row in charging_stations_df.iterrows():
            station_intersection = row['pStationIntersection']
            station_to_index[station_intersection] = idx
        
        # Update charging prices for stations in the dictionary
        for station, price in charging_prices.items():
            if station in station_to_index:
                row_idx = station_to_index[station]
                charging_stations_df.loc[row_idx, 'pChargingPrice'] = price
                if verbose >= 1:
                    print(f"Updated charging price for station {station} to {price}")
            else:
                print(f"Warning: Charging station {station} not found in data, skipping price update")

    # Extract list of unique EVs from delivery points
    evs = sorted(delivery_points_df["EV"].unique().tolist())

    # Return the raw data in a structured format
    map_data = {
        'unindexed_df': unindexed_df,
        'paths_df': paths_df,
        'delivery_points_df': delivery_points_df,
        'charging_stations_df': charging_stations_df,
        'time_periods_df': time_periods_df,
        'coordinates': coordinates,
        'clean_column_name': clean_column_name,
        'evs': evs
    }

    return map_data


def extract_electricity_costs(map_data: dict) -> dict:
    """
    Extract electricity costs from map data in the format expected by regression models.
    
    Parameters
    ----------
    map_data: dict
        The raw map data returned by load_excel_map_data().
    
    Returns
    -------
    electricity_costs: dict
        Dictionary with format {time_period: electricity_cost} or empty dict if no data available.
    """
    time_periods_df = map_data.get('time_periods_df')
    
    if time_periods_df is None:
        return {}
    
    if 'pPeriod' not in time_periods_df.columns or 'pElectricityCost' not in time_periods_df.columns:
        return {}
    
    electricity_costs = {}
    for _, row in time_periods_df.iterrows():
        period = int(row['pPeriod'])
        cost = float(row['pElectricityCost'])
        electricity_costs[period] = cost
    
    return electricity_costs


def filter_map_data_for_ev(map_data: dict, ev: int) -> dict:
    """
    Filter map data for a specific EV and convert to Pyomo input format.

    Parameters
    ----------
    map_data: dict
        The raw map data returned by load_excel_map_data().
    ev: int
        The specific EV to filter delivery points for.

    Returns
    -------
    input_data: dict
        The input data for the routing_model in the format required by Pyomo, including coordinates.
    """

    # Extract dataframes from map_data
    unindexed_df = map_data['unindexed_df']
    paths_df = map_data['paths_df']
    delivery_points_df = map_data['delivery_points_df']
    charging_stations_df = map_data['charging_stations_df']
    coordinates = map_data['coordinates']
    clean_column_name = map_data['clean_column_name']

    # Filter delivery points by EV
    delivery_points_df = delivery_points_df[delivery_points_df["EV"] == ev]

    # Extract sets
    # Get all unique intersections from the paths dataframe
    all_intersections = paths_df["pOriginIntersection"].tolist() + paths_df["pDestinationIntersection"].tolist()
    intersections = sorted(list(set(all_intersections)))
    
    # Path IDs from 1 to number of paths
    paths = list(range(1, len(paths_df) + 1))
    
    # Delivery points and charging stations are defined by their respective sheets
    delivery_points = delivery_points_df["pDeliveryIntersection"].tolist()
    charging_stations = charging_stations_df["pStationIntersection"].tolist()

    # Start building the input data dictionary
    input_data = {None: {
        'sIntersections': {None: intersections},
        'sPaths': {None: paths},
        'sDeliveryPoints': {None: delivery_points},
        'sChargingStations': {None: charging_stations},
        'coordinates': coordinates,  # Add coordinates to input_data
    }}

    # Process unindexed parameters (scalar values)
    for idx, row in unindexed_df.iterrows():
        param_name = idx  # idx is already cleaned in load_excel_map_data
        value = row["Value"].item()
        input_data[None][param_name] = {None: value}
    input_data[None]['pNumIntersections'] = {None: len(intersections)}

    # Process indexed parameters for paths, delivery points and charging stations
    for df, points_list in [
        (paths_df, paths),
        (delivery_points_df, delivery_points),
        (charging_stations_df, charging_stations)
    ]:
        for col in df.columns:
            param_data = {point: getattr(row, col) for point, row in zip(points_list, df.itertuples(index=False))}
            input_data[None][col] = param_data

    # Create pPath parameter: mapping from (origin, destination) to path ID
    pPath_data = {}
    for path_id, row in zip(paths, paths_df.itertuples(index=False)):
        origin = row.pOriginIntersection
        destination = row.pDestinationIntersection
        pPath_data[(origin, destination)] = path_id
    input_data[None]['pPath'] = pPath_data

    return input_data

