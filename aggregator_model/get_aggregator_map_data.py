"""
This function handles the input data of the aggregator model,
loading the map data from Excel file
"""

import pandas as pd


def load_aggregator_excel_data(file_path: str, verbose: int = 0) -> dict:
    """
    Load data from an Excel file for the aggregator optimization model.

    Parameters
    ----------
    file_path: str
        The path to the Excel file (.xlsx) containing the aggregator data.
    verbose: int
        Verbosity level.

    Returns
    -------
    input_data: dict
        The input data for the model in the format required by Pyomo.
    """

    # Read sheets from the Excel file
    charging_stations_df = pd.read_excel(file_path, sheet_name="sChargingStations")

    # Function to clean column names by taking only the first word
    def clean_column_name(col_name):
        return col_name.split(" ")[0]

    # Clean column names for all dataframes
    charging_stations_df.columns = [clean_column_name(col) for col in charging_stations_df.columns]

    if verbose >= 1:
        print("Loaded charging stations:", charging_stations_df['pStationIntersection'].tolist())

    # Extract sets
    charging_stations = charging_stations_df["pStationIntersection"].tolist()

    # Start building the input data dictionary
    input_data = {None: {
        'sChargingStations': {None: charging_stations},
    }}

    # Process indexed parameters for charging stations
    for col in charging_stations_df.columns:
        if col != 'pStationIntersection':  # Skip the index column
            param_data = {station: getattr(row, col) 
                         for station, row in zip(charging_stations, charging_stations_df.itertuples(index=False))}
            input_data[None][col] = param_data

    return input_data
