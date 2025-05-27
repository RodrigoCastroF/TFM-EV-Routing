import pandas as pd


def load_excel_map_data(file_path: str) -> dict:
    """
    Load data from an Excel file for the EV routing optimization model.

    Parameters
    ----------
    file_path: str
        The path to the Excel file (.xlsx) containing the data.

    Returns
    -------
    map_data: dict
        The raw map data loaded from Excel, containing all dataframes, metadata, and list of EVs.
    """

    # Read sheets from the Excel file
    unindexed_df = pd.read_excel(file_path, sheet_name="Unindexed", index_col="Name")
    paths_df = pd.read_excel(file_path, sheet_name="sPaths")
    delivery_points_df = pd.read_excel(file_path, sheet_name="sDeliveryPoints")
    charging_stations_df = pd.read_excel(file_path, sheet_name="sChargingStations")

    # Function to clean column names by taking only the first word
    def clean_column_name(col_name):
        return col_name.split(" ")[0]

    # Clean column names for all dataframes
    for df in [paths_df, delivery_points_df, charging_stations_df]:
        df.columns = [clean_column_name(col) for col in df.columns]

    # Extract list of unique EVs from delivery points
    evs = sorted(delivery_points_df["EV"].unique().tolist())

    # Return the raw data in a structured format
    map_data = {
        'unindexed_df': unindexed_df,
        'paths_df': paths_df,
        'delivery_points_df': delivery_points_df,
        'charging_stations_df': charging_stations_df,
        'clean_column_name': clean_column_name,
        'evs': evs
    }

    return map_data


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
        The input data for the model in the format required by Pyomo.
    """

    # Extract dataframes from map_data
    unindexed_df = map_data['unindexed_df']
    paths_df = map_data['paths_df']
    delivery_points_df = map_data['delivery_points_df']
    charging_stations_df = map_data['charging_stations_df']
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
    }}

    # Process unindexed parameters (scalar values)
    for idx, row in unindexed_df.iterrows():
        param_name = clean_column_name(idx)
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

