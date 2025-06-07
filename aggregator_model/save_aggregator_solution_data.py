"""
These functions extract and save the solution for the simplified aggregator model
"""

import pandas as pd
import pyomo.environ as pyo


def extract_aggregator_solution_data(model_instance, charging_stations, min_prices, max_prices):
    """
    Extract solution data from a solved Pyomo aggregator model instance.

    Parameters
    ----------
    model_instance: pyomo.core.base.PyomoModel.ConcreteModel
        A solved Pyomo model instance containing the solution values.
    charging_stations: list
        List of charging station IDs.
    min_prices: dict
        Dictionary of minimum charging prices by station.
    max_prices: dict
        Dictionary of maximum charging prices by station.

    Returns
    -------
    solution_data: dict
        Dictionary containing the solution data in structured format.
    """

    # Helper function to safely get variable value
    def get_var_value(var):
        """Safely extract variable value, returning None if not available."""
        try:
            return var.value
        except (ValueError, AttributeError):
            return None

    # Create charging stations solution data
    charging_stations_data = []
    
    for station in charging_stations:
        var_name = f'rc_{station}'
        charging_price_var = getattr(model_instance, 'x', None)
        price_value = None
        if charging_price_var and var_name in charging_price_var:
            price_value = get_var_value(charging_price_var[var_name])
        
        row_data = {
            'pStationIntersection': station,
            'pMinChargingPrice': min_prices[station],
            'pMaxChargingPrice': max_prices[station],
            'vChargingPrice': price_value,
        }
        charging_stations_data.append(row_data)
    
    charging_stations_df = pd.DataFrame(charging_stations_data)
    
    solution_data = {
        'charging_stations_df': charging_stations_df
    }

    return solution_data


def save_aggregator_solution_data(solution_data, file_path: str):
    """
    Save the simplified aggregator solution data to an Excel file.

    Parameters
    ----------
    solution_data: dict
        A dictionary containing the solution data from extract_aggregator_solution_data().
    file_path: str
        The path where the Excel file should be saved (should end with .xlsx).
    """
    
    charging_stations_df = solution_data['charging_stations_df']
    
    # Save to Excel file
    with pd.ExcelWriter(file_path, engine='openpyxl') as writer:
        charging_stations_df.to_excel(writer, sheet_name='ChargingStations', index=False)
