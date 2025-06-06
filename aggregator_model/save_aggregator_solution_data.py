"""
These functions extract and save the solution for the aggregator model
"""

import pandas as pd
import pyomo.environ as pyo


def extract_aggregator_solution_data(model_instance):
    """
    Extract solution data from a solved Pyomo aggregator model instance.

    Parameters
    ----------
    model_instance: pyomo.core.base.PyomoModel.ConcreteModel
        A solved Pyomo model instance containing the solution values.

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
    
    for station in model_instance.sChargingStations:
        row_data = {
            'pStationIntersection': station,
            'pMinChargingPrice': model_instance.pMinChargingPrice[station],
            'pMaxChargingPrice': model_instance.pMaxChargingPrice[station],
            'vChargingPrice': get_var_value(model_instance.vChargingPrice[station]),
        }
        charging_stations_data.append(row_data)
    
    charging_stations_df = pd.DataFrame(charging_stations_data)
    
    # Create time periods data
    time_periods_data = []
    
    for period in model_instance.sTimePeriods:
        row_data = {
            'pPeriod': period,
            'pElectricityCost': model_instance.pElectricityCost[period],
        }
        time_periods_data.append(row_data)
    
    time_periods_df = pd.DataFrame(time_periods_data)
    
    # Create demand data (station-period combinations)
    demand_data = []
    
    for station in model_instance.sChargingStations:
        for period in model_instance.sTimePeriods:
            row_data = {
                'pStationIntersection': station,
                'pPeriod': period,
                'vAggregatedDemand': get_var_value(model_instance.vAggregatedDemand[station, period]),
            }
            demand_data.append(row_data)
    
    demand_df = pd.DataFrame(demand_data)
    
    # Calculate revenue and cost breakdown
    revenue_data = []
    
    for station in model_instance.sChargingStations:
        for period in model_instance.sTimePeriods:
            charging_price = get_var_value(model_instance.vChargingPrice[station])
            aggregated_demand = get_var_value(model_instance.vAggregatedDemand[station, period])
            electricity_cost = model_instance.pElectricityCost[period]
            
            if charging_price is not None and aggregated_demand is not None:
                revenue = charging_price * aggregated_demand
                cost = electricity_cost * aggregated_demand
                profit = revenue - cost
            else:
                revenue = cost = profit = None
            
            row_data = {
                'pStationIntersection': station,
                'pPeriod': period,
                'vChargingPrice': charging_price,
                'vAggregatedDemand': aggregated_demand,
                'pElectricityCost': electricity_cost,
                'Revenue': revenue,
                'Cost': cost,
                'Profit': profit,
            }
            revenue_data.append(row_data)
    
    revenue_df = pd.DataFrame(revenue_data)
    
    solution_data = {
        'charging_stations_df': charging_stations_df,
        'time_periods_df': time_periods_df,
        'demand_df': demand_df,
        'revenue_df': revenue_df
    }

    return solution_data


def save_aggregator_solution_data(solution_data, file_path: str):
    """
    Save the aggregator solution data to an Excel file.

    Parameters
    ----------
    solution_data: dict
        A dictionary containing the solution data from extract_aggregator_solution_data().
    file_path: str
        The path where the Excel file should be saved (should end with .xlsx).
    """
    
    charging_stations_df = solution_data['charging_stations_df']
    time_periods_df = solution_data['time_periods_df']
    demand_df = solution_data['demand_df']
    revenue_df = solution_data['revenue_df']
    
    # Save to Excel file with multiple sheets
    with pd.ExcelWriter(file_path, engine='openpyxl') as writer:
        charging_stations_df.to_excel(writer, sheet_name='ChargingStations', index=False)
        time_periods_df.to_excel(writer, sheet_name='TimePeriods', index=False)
        demand_df.to_excel(writer, sheet_name='Demand', index=False)
        revenue_df.to_excel(writer, sheet_name='RevenueBreakdown', index=False)
