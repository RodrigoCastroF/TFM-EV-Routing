"""
These functions extract and represent the solution for a single EV
"""


import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from pathlib import Path


def extract_solution_data(model_instance):
    """
    Extract solution data from a solved Pyomo routing_model instance.

    Parameters
    ----------
    model_instance: pyomo.core.base.PyomoModel.ConcreteModel
        A solved Pyomo routing_model instance containing the solution values.
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
        row_data = {
            'intersection': intersection,
            'v01VisitIntersection': get_var_value(model_instance.v01VisitIntersection[intersection]),
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
        row_data = {
            'pOriginIntersection': model_instance.pOriginIntersection[path],
            'pDestinationIntersection': model_instance.pDestinationIntersection[path],
            'v01TravelPath': get_var_value(model_instance.v01TravelPath[path]),
        }

        # Add auxiliary variables if they exist (for linearized constraints)
        if hasattr(model_instance, 'vXiSoC'):
            row_data['vXiSoC'] = get_var_value(model_instance.vXiSoC[path])
        else:
            row_data['vXiSoC'] = None

        if hasattr(model_instance, 'vZetaTime'):
            row_data['vZetaTime'] = get_var_value(model_instance.vZetaTime[path])
        else:
            row_data['vZetaTime'] = None

        paths_data.append(row_data)
    
    paths_df = pd.DataFrame(paths_data)
    
    solution_data = {
        'intersections_df': intersections_df,
        'paths_df': paths_df
    }

    return solution_data


def save_solution_data(solution_data, file_path: str, metadata=None):
    """
    Save the solution data from a solved Pyomo routing_model instance to an Excel file.

    Parameters
    ----------
    solution_data: dict
        A dictionary containing the solution data from extract_solution_data().
    file_path: str
        The path where the Excel file should be saved (should end with .xlsx).
    metadata: dict, optional
        Dictionary containing solver metadata (solver_status, termination_condition, etc.).
    """
    
    intersections_df = solution_data['intersections_df']
    paths_df = solution_data['paths_df']
    
    # Save to Excel file
    with pd.ExcelWriter(file_path, engine='openpyxl') as writer:
        intersections_df.to_excel(writer, sheet_name='sIntersections', index=False)
        paths_df.to_excel(writer, sheet_name='sPaths', index=False)
        
        # Save metadata if provided
        if metadata:
            metadata_df = pd.DataFrame([metadata])
            metadata_df.to_excel(writer, sheet_name='Unindexed', index=False)


def load_solution_data(file_path: str):
    """
    Load solution data from an Excel file saved by save_solution_data.

    Parameters
    ----------
    file_path: str
        The path to the Excel file to load.

    Returns
    -------
    tuple
        (solution_data, metadata) where solution_data is a dict with 'intersections_df' and 'paths_df',
        and metadata is a dict with solver information (or None if not available).
    """
    
    # Load the main solution data
    intersections_df = pd.read_excel(file_path, sheet_name='sIntersections')
    paths_df = pd.read_excel(file_path, sheet_name='sPaths')
    
    solution_data = {
        'intersections_df': intersections_df,
        'paths_df': paths_df
    }
    
    # Try to load metadata
    metadata = None
    try:
        metadata_df = pd.read_excel(file_path, sheet_name='Unindexed')
        if not metadata_df.empty:
            metadata = metadata_df.iloc[0].to_dict()
    except Exception:
        # Metadata sheet doesn't exist or couldn't be read
        pass
    
    return solution_data, metadata


def create_solution_map(solution_data, input_data, file_path: str, ev: int = 1, eps: float = 1e-5, decimal_precision: int = 1):
    """
    Create a visual map representation of the routing solution.
    
    Parameters
    ----------
    solution_data: dict
        A dictionary containing the solution data from extract_solution_data.
    input_data: dict
        The input data dictionary from filter_map_data_for_ev, which includes coordinates.
    file_path: str
        The path where the image should be saved (should end with .png).
    ev: int
        The EV number for which to show delivery points.
    eps: float
        Tolerance for considering arrival and departure values as identical.
    decimal_precision: int
        Number of decimal places to show for time and SoC values.
    """
    
    # Extract data
    paths_data = input_data[None]
    intersections = paths_data['sIntersections'][None]
    paths = paths_data['sPaths'][None]
    delivery_points = paths_data['sDeliveryPoints'][None]
    charging_stations = paths_data['sChargingStations'][None]
    coordinates = paths_data.get('coordinates')
    
    # Extract solution data
    intersections_df = solution_data['intersections_df']
    paths_df = solution_data['paths_df']

    # Consider only visited intersections and paths
    # Note Gurobi may save the binary variable as 0.999999936,
    # so we check for `abs(intersections_df['v01VisitIntersection'] - 1) < eps`
    # instead of `intersections_df['v01VisitIntersection'] == 1
    intersections_df = intersections_df[abs(intersections_df['v01VisitIntersection'] - 1) < eps]
    paths_df = paths_df[abs(paths_df['v01TravelPath'] - 1) < eps]
    
    # Create graph
    G = nx.Graph()
    
    # Add nodes
    for intersection in intersections:
        G.add_node(intersection)
    
    # Add edges with path types
    edge_types = {}
    for path_id in paths:
        origin = paths_data['pOriginIntersection'][path_id]
        dest = paths_data['pDestinationIntersection'][path_id]
        path_type = paths_data['pTypePath'][path_id]
        
        # Only add edge once for bidirectional paths
        edge = tuple(sorted([origin, dest]))
        if edge not in edge_types:
            G.add_edge(origin, dest)
            edge_types[edge] = path_type
    
    # Try to use coordinates from input_data, fallback to automatic layout
    if coordinates:
        # Use coordinates from input_data
        pos = {}
        for node_id, (x, y) in coordinates.items():
            if node_id in intersections:
                pos[node_id] = (x, y)
        
        # Normalize coordinates to fit in a reasonable range
        if pos:
            x_coords = [x for x, y in pos.values()]
            y_coords = [y for x, y in pos.values()]
            x_min, x_max = min(x_coords), max(x_coords)
            y_min, y_max = min(y_coords), max(y_coords)
            
            # Scale to fit in [-1, 1] range
            x_range = x_max - x_min if x_max != x_min else 1
            y_range = y_max - y_min if y_max != y_min else 1
            
            for node in pos:
                x, y = pos[node]
                pos[node] = (2 * (x - x_min) / x_range - 1, 2 * (y - y_min) / y_range - 1)
        
        print("Using predefined node coordinates from input data")
    else:
        # Fallback to automatic layout
        print("No predefined coordinates found, using automatic layout")
        np.random.seed(42)
        try:
            pos = nx.planar_layout(G, scale=2)
        except:
            try:
                pos = nx.kamada_kawai_layout(G, scale=2)
            except:
                pos = nx.spring_layout(G, k=5, iterations=200, seed=42, scale=2)
    
    # Create figure
    plt.figure(figsize=(16, 12))
    plt.axis('off')
    
    # Draw edges by type
    for edge, path_type in edge_types.items():
        x_coords = [pos[edge[0]][0], pos[edge[1]][0]]
        y_coords = [pos[edge[0]][1], pos[edge[1]][1]]
        
        if path_type == "Main Type 1":
            plt.plot(x_coords, y_coords, 'k-', linewidth=4, alpha=0.7)
        elif path_type == "Main Type 2":
            # Double line effect
            plt.plot(x_coords, y_coords, 'k-', linewidth=6, alpha=0.7)
            plt.plot(x_coords, y_coords, 'w-', linewidth=2, alpha=0.9)
        else:  # Secondary
            plt.plot(x_coords, y_coords, 'k-', linewidth=1, alpha=0.7)
    
    # Draw solution paths in red
    for _, row in paths_df.iterrows():
        origin = row['pOriginIntersection']
        dest = row['pDestinationIntersection']
        x_coords = [pos[origin][0], pos[dest][0]]
        y_coords = [pos[origin][1], pos[dest][1]]
        plt.plot(x_coords, y_coords, 'r-', linewidth=6, alpha=0.8)
    
    # Get start/end points from unindexed parameters
    start_point = paths_data['pStartingPoint'][None]
    end_point = paths_data['pEndingPoint'][None]
    
    # Handle case where start and end points are different but at same coordinates
    same_physical_location = False
    modified_pos = pos.copy()  # Create a copy to modify positions if needed
    
    if start_point != end_point and start_point in pos and end_point in pos:
        start_coords = pos[start_point]
        end_coords = pos[end_point]
        # Check if they have the same coordinates (within tolerance)
        if abs(start_coords[0] - end_coords[0]) < 1e-6 and abs(start_coords[1] - end_coords[1]) < 1e-6:
            same_physical_location = True
            # Create a small offset for both points - center them around the original location
            offset = 0.025  # Half the previous offset since we're splitting the distance
            modified_pos[start_point] = (start_coords[0], start_coords[1] + offset)  # Start point above
            modified_pos[end_point] = (end_coords[0], end_coords[1] - offset)  # End point below
    
    # Draw nodes
    start_end_points = [start_point, end_point] if start_point != end_point else [start_point]
    regular_nodes = [n for n in intersections if n not in delivery_points and n not in charging_stations and n not in start_end_points]
    delivery_only = [n for n in delivery_points if n not in start_end_points]
    
    # Regular intersections
    nx.draw_networkx_nodes(G, modified_pos, nodelist=regular_nodes, node_color='lightgray', 
                          node_size=500, alpha=0.8, edgecolors='black', linewidths=0.5)
    
    # Charging stations (exclude start/end if they are also charging stations)
    charging_only = [n for n in charging_stations if n not in start_end_points]
    nx.draw_networkx_nodes(G, modified_pos, nodelist=charging_only, node_color='lightblue', 
                          node_size=500, alpha=0.9, node_shape='s', edgecolors='black', linewidths=1)
    
    # Delivery points (exclude start/end)
    nx.draw_networkx_nodes(G, modified_pos, nodelist=delivery_only, node_color='orange', 
                          node_size=500, alpha=0.9, edgecolors='black', linewidths=1)
    
    # Start/End points - handle them separately if they're at the same location
    if start_point != end_point and same_physical_location:
        # Draw start point
        nx.draw_networkx_nodes(G, modified_pos, nodelist=[start_point], node_color='red', 
                              node_size=500, alpha=0.9, edgecolors='red', linewidths=2)
        # Draw end point with different color to distinguish
        nx.draw_networkx_nodes(G, modified_pos, nodelist=[end_point], node_color='green', 
                              node_size=500, alpha=0.9, edgecolors='green', linewidths=2)
    else:
        # Original behavior for other cases
        nx.draw_networkx_nodes(G, modified_pos, nodelist=start_end_points, node_color='red', 
                              node_size=500, alpha=0.9, edgecolors='darkred', linewidths=2)
    
    # Add labels for intersections
    nx.draw_networkx_labels(G, modified_pos, font_size=12, font_weight='bold')
    
    # Add direction arrow for the start node (showing first path direction)
    if start_point in modified_pos:
        # Find the first path that starts from the start point
        first_path = None
        for _, row in paths_df.iterrows():
            if row['pOriginIntersection'] == start_point:
                first_path = row
                break
        
        if first_path is not None:
            dest = first_path['pDestinationIntersection']
            if dest in modified_pos:
                # Calculate direction vector using REAL physical locations (not modified positions)
                start_pos_real = pos[start_point]  # Use original position for direction calculation
                dest_pos_real = pos[dest] if dest in pos else modified_pos[dest]  # Use original dest position if available
                
                # Direction vector from real start to real destination
                dx = dest_pos_real[0] - start_pos_real[0]
                dy = dest_pos_real[1] - start_pos_real[1]
                
                # Use modified position for arrow placement but real position for direction
                start_pos_visual = modified_pos[start_point]
                
                # Normalize the direction vector
                length = np.sqrt(dx**2 + dy**2)
                if length > 0:
                    dx_norm = dx / length
                    dy_norm = dy / length
                    
                    # Position arrow close to the start node, slightly higher up
                    arrow_offset = 0.05  # Reduced distance - closer to the node center
                    arrow_start_x = start_pos_visual[0] + dx_norm * arrow_offset
                    arrow_start_y = start_pos_visual[1] + dy_norm * arrow_offset + 0.02  # Slightly higher up
                    
                    # Much bigger arrow length (2x size)
                    arrow_length = 0.08  # 2x bigger than the previous 0.04
                    arrow_end_x = arrow_start_x + dx_norm * arrow_length
                    arrow_end_y = arrow_start_y + dy_norm * arrow_length
                    
                    # Draw arrow
                    plt.annotate('', xy=(arrow_end_x, arrow_end_y), xytext=(arrow_start_x, arrow_start_y),
                               arrowprops=dict(arrowstyle='->', color='red', lw=2, alpha=0.8))
    
    # Add solution information for all visited nodes
    for _, row in intersections_df.iterrows():
        intersection = row['intersection']
        
        # Get solution values from the DataFrame
        time_arr = row['vTimeArrival']
        time_dep = row['vTimeDeparture']
        soc_arr = row['vSoCArrival']
        soc_dep = row['vSoCDeparture']
        
        # Format text more compactly
        info_text = []
        if time_arr is not None and time_dep is not None:
            if abs(time_arr - time_dep) < eps:  # Values are essentially identical
                info_text.append(f"Time: {time_arr:.{decimal_precision}f}")
            else:
                info_text.append(f"Time: {time_arr:.{decimal_precision}f}→{time_dep:.{decimal_precision}f}")
        if soc_arr is not None and soc_dep is not None:
            if abs(soc_arr - soc_dep) < eps:  # Values are essentially identical
                info_text.append(f"SoC: {soc_arr:.{decimal_precision}f}")
            else:
                info_text.append(f"SoC: {soc_arr:.{decimal_precision}f}→{soc_dep:.{decimal_precision}f}")
        
        # Add objective function components
        if intersection in delivery_points:
            # Delivery time (parameter pTimeWithoutPenalty)
            delivery_time = paths_data['pTimeWithoutPenalty'][intersection]
            info_text.append(f"Delivery Time: {delivery_time:.{decimal_precision}f}")
            
            # Delay penalty calculation: pDelayPenalty * vTimeDelay
            delay_time = row['vTimeDelay'] if row['vTimeDelay'] is not None else 0
            delay_penalty_rate = paths_data['pDelayPenalty'][intersection]
            delay_penalty = delay_penalty_rate * delay_time
            if delay_penalty > eps:
                info_text.append(f"Delay Penalty: {delay_penalty:.{decimal_precision}f}")
            else:
                info_text.append("Delay Penalty: 0")
        
        if intersection in charging_stations:
            # Charging cost calculation: pChargingPrice * pChargingPower * vTimeCharging * pChargerEfficiencyRate
            charging_time = row['vTimeCharging'] if row['vTimeCharging'] is not None else 0
            if charging_time > eps:  # Only show if actually charging
                charging_price = paths_data['pChargingPrice'][intersection]
                charging_power = paths_data['pChargingPower'][intersection]
                charger_efficiency = paths_data['pChargerEfficiencyRate'][intersection]
                charging_cost = charging_price * charging_power * charging_time * charger_efficiency
                info_text.append(f"Charging Cost: {charging_cost:.{decimal_precision}f}")
            else:
                info_text.append("Charging Cost: 0")
        
        if info_text:
            # Special positioning for start/end nodes at same physical location
            if same_physical_location and intersection in [start_point, end_point]:
                if intersection == start_point:
                    # Position start point label to the right and slightly up
                    xytext_offset = (20, 10)
                else:  # end_point
                    # Position end point label to the right and slightly down
                    xytext_offset = (20, -15)
            else:
                # Default positioning for other nodes
                xytext_offset = (15, 15)
            
            plt.annotate('\n'.join(info_text), 
                       xy=modified_pos[intersection], xytext=xytext_offset,
                       textcoords='offset points', fontsize=10, fontweight='bold',
                       bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.9, edgecolor='gray'))
    
    # Calculate and display total objective function value
    total_charging_cost = 0
    total_delay_penalty = 0
    
    for _, row in intersections_df.iterrows():
        intersection = row['intersection']
        
        # Sum charging costs
        if intersection in charging_stations:
            charging_time = row['vTimeCharging'] if row['vTimeCharging'] is not None else 0
            if charging_time > eps:
                charging_price = paths_data['pChargingPrice'][intersection]
                charging_power = paths_data['pChargingPower'][intersection]
                charger_efficiency = paths_data['pChargerEfficiencyRate'][intersection]
                charging_cost = charging_price * charging_power * charging_time * charger_efficiency
                total_charging_cost += charging_cost
        
        # Sum delay penalties
        if intersection in delivery_points:
            delay_time = row['vTimeDelay'] if row['vTimeDelay'] is not None else 0
            delay_penalty_rate = paths_data['pDelayPenalty'][intersection]
            delay_penalty = delay_penalty_rate * delay_time
            total_delay_penalty += delay_penalty
    
    total_objective = total_charging_cost + total_delay_penalty
    
    # Add legend
    legend_elements = [
        plt.Line2D([0], [0], color='black', linewidth=4, label='Main Type 1'),
        plt.Line2D([0], [0], color='black', linewidth=6, label='Main Type 2'),
        plt.Line2D([0], [0], color='black', linewidth=1, label='Secondary'),
        plt.Line2D([0], [0], color='red', linewidth=6, label='Solution Path'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='orange', markersize=8, label='Delivery Points'),
        plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='lightblue', markersize=8, label='Charging Stations')
    ]
    
    # Add start/end legend elements based on whether they're at same location
    if start_point != end_point and same_physical_location:
        legend_elements.extend([
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10, label='Start Point'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=10, label='End Point')
        ])
    else:
        legend_elements.append(
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10, label='Start/End')
        )
    
    plt.legend(handles=legend_elements, loc='upper right')
    
    # Update title to include objective function value
    plt.title(f'EV Routing Solution - EV {ev}\n'
              f'Total Cost: {total_objective:.{decimal_precision}f} '
              f'(Charging: {total_charging_cost:.{decimal_precision}f}, '
              f'Delay Penalty: {total_delay_penalty:.{decimal_precision}f})', 
              fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    # Save image
    output_path = Path(file_path)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
