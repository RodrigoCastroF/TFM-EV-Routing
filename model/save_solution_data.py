import pandas as pd


def save_solution_data(model_instance, file_path: str):
    """
    Save the solution data from a solved Pyomo model instance to an Excel file.

    Parameters
    ----------
    model_instance: pyomo.core.base.PyomoModel.ConcreteModel
        A solved Pyomo model instance containing the solution values.
    file_path: str
        The path where the Excel file should be saved (should end with .xlsx).
    """
    
    # Helper function to safely get variable value
    def get_var_value(var):
        """Safely extract variable value, returning None if not available."""
        try:
            return var.value
        except (ValueError, AttributeError):
            return None
    
    # Create sIntersections sheet
    intersections_data = []
    
    for intersection in model_instance.sIntersections:
        # Get the visit intersection value
        visit_value = get_var_value(model_instance.v01VisitIntersection[intersection])
        
        # Only include rows where v01VisitIntersection is not 0
        if visit_value is not None and visit_value != 0:
            row_data = {
                'intersection': intersection,
                'v01VisitIntersection': visit_value,
                'vSoCArrival': get_var_value(model_instance.vSoCArrival[intersection]),
                'vSoCDeparture': get_var_value(model_instance.vSoCDeparture[intersection]),
                'vTimeArrival': get_var_value(model_instance.vTimeArrival[intersection]),
                'vTimeDeparture': get_var_value(model_instance.vTimeDeparture[intersection]),
            }
            
            # Add charging station variables if the intersection is a charging station
            if intersection in model_instance.sChargingStations:
                row_data['v01Charge'] = get_var_value(model_instance.v01Charge[intersection])
                row_data['vTimeCharging'] = get_var_value(model_instance.vTimeCharging[intersection])
            else:
                row_data['v01Charge'] = None
                row_data['vTimeCharging'] = None
                
            # Add delivery point variable if the intersection is a delivery point
            if intersection in model_instance.sDeliveryPoints:
                row_data['vTimeDelay'] = get_var_value(model_instance.vTimeDelay[intersection])
            else:
                row_data['vTimeDelay'] = None
                
            intersections_data.append(row_data)
    
    intersections_df = pd.DataFrame(intersections_data)
    
    # Create sPaths sheet
    paths_data = []
    
    for path in model_instance.sPaths:
        # Get the travel path value
        travel_value = get_var_value(model_instance.v01TravelPath[path])
        
        # Only include rows where v01TravelPath is not 0
        if travel_value is not None and travel_value != 0:
            row_data = {
                'pOriginIntersection': model_instance.pOriginIntersection[path],
                'pDestinationIntersection': model_instance.pDestinationIntersection[path],
                'v01TravelPath': travel_value,
            }
            paths_data.append(row_data)
    
    paths_df = pd.DataFrame(paths_data)
    
    # Save to Excel file
    with pd.ExcelWriter(file_path, engine='openpyxl') as writer:
        intersections_df.to_excel(writer, sheet_name='sIntersections', index=False)
        paths_df.to_excel(writer, sheet_name='sPaths', index=False)
