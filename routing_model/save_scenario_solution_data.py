"""
These functions extract and represent the solution for an entire scenario (all EVs)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def extract_aggregated_demand(all_ev_results, map_data, eps=1e-5, verbose=0):
    """
    Extract charging demand by time period for all EVs in a scenario using the time period adaptation algorithm.
    
    Args:
        all_ev_results: Dictionary returned by solve_for_all_evs function with format:
                       {ev_id: {'ev': ev_id, 'solution_data': {...}, ...}, ...}
        map_data: Original map data dictionary returned by load_excel_map_data, 
                 containing charging station parameters
        eps: Tolerance for considering binary variables as 1 (default: 1e-5)
        verbose: Level of verbosity (0: no print, 1+: print details)
        
    Returns:
        pandas.DataFrame: DataFrame with columns [charging_station, time_period, aggregated_demand]
                         where time_period ranges from 0-23 (representing hours 00:00-00:59, 01:00-01:59, etc.)
                         and aggregated_demand is in kWh
    """
    
    # Get charging station information from map_data
    charging_stations_df = map_data['charging_stations_df']
    
    # Create a mapping from charging station intersection to charging power
    station_power_map = {}
    for _, row in charging_stations_df.iterrows():
        station_intersection = row['pStationIntersection']
        charging_power = row['pChargingPower']  # kW
        station_power_map[station_intersection] = charging_power
    
    # Get all charging stations
    all_charging_stations = list(station_power_map.keys())
    
    # Initialize results storage: A[v,i,t] for each EV v, charging station i, time period t
    charging_times = {}  # {(ev, station, time_period): charging_time_hours}
    
    # Process each EV's solution
    for ev_id, ev_results in all_ev_results.items():
        # Skip if this EV didn't solve successfully or doesn't have solution data
        if 'solution_data' not in ev_results:
            continue
            
        solution_data = ev_results['solution_data']
        intersections_df = solution_data['intersections_df']
        
        # Filter for charging stations where the EV actually charges
        # (v01Charge close to 1, considering solver precision issues)
        charging_stations_visited = intersections_df[
            (intersections_df['intersection'].isin(all_charging_stations)) &
            (abs(intersections_df['v01Charge'] - 1) < eps)
        ]
        
        # Process each charging station visited by this EV
        for _, row in charging_stations_visited.iterrows():
            station = row['intersection']
            t_arrival = row['vTimeArrival']  # hours from 00:00
            t_departure = row['vTimeDeparture']  # hours from 00:00
            
            # Skip if arrival/departure times are not available
            if pd.isna(t_arrival) or pd.isna(t_departure):
                continue
                
            # Calculate charging time for each time period t=0,1,2,...,23
            for t in range(24):  # time periods 0-23
                # Apply the formula: A_{v,i,t} = max{0, min{t+1, t_departure} - max{t, t_arrival}}
                charging_time = max(0, min(t + 1, t_departure) - max(t, t_arrival))
                
                if charging_time > eps:  # Only store non-zero charging times
                    charging_times[(ev_id, station, t)] = charging_time
    
    # Calculate aggregated demand for each (charging_station, time_period) combination
    aggregated_results = []
    
    for station in all_charging_stations:
        station_power = station_power_map[station]  # kW
        
        for t in range(24):  # time periods 0-23
            # Sum charging times across all EVs for this station and time period
            total_charging_time = 0
            for ev_id in all_ev_results.keys():
                charging_time = charging_times.get((ev_id, station, t), 0)
                total_charging_time += charging_time
            
            # Calculate aggregated demand: P^C_i * sum_v A_{v,i,t} [kWh]
            aggregated_demand = station_power * total_charging_time
            
            aggregated_results.append({
                'charging_station': station,
                'time_period': t,
                'aggregated_demand': aggregated_demand
            })
    
    # Convert to DataFrame and sort for better readability
    result_df = pd.DataFrame(aggregated_results)
    result_df = result_df.sort_values(['charging_station', 'time_period']).reset_index(drop=True)
    
    if verbose >= 1:
        print("Raw aggregated demand results:")
        print(result_df.to_string())
    
    return result_df


def create_scenario_analysis_plots(all_ev_results, map_data, file_path: str, aggregated_demand: pd.DataFrame = None, eps: float = 1e-5, verbose: int = 0):
    """
    Create a figure with individual plots for each EV trajectory and one plot for aggregated demand.
    
    Args:
        all_ev_results: Dictionary returned by solve_for_all_evs function with format:
                       {ev_id: {'ev': ev_id, 'solution_data': {...}, ...}, ...}
        map_data: Original map data dictionary returned by load_excel_map_data, 
                 containing charging station parameters and delivery points
        file_path: str - Path where the figure should be saved (should end with .png)
        aggregated_demand: pd.DataFrame - Aggregated demand data from extract_aggregated_demand function
                          If None, it will be calculated
        eps: float - Tolerance for considering binary variables as 1 (default: 1e-5)
        verbose: int - Level of verbosity (0: no print, 1+: print details)
    """
    
    # Get starting time from unindexed parameters
    starting_time = map_data['unindexed_df'].loc['pStartingTime', 'Value']
    # Get max time from unindexed parameters
    max_time = map_data['unindexed_df'].loc['pMaxTime', 'Value']
    
    # Get charging stations and delivery points from map_data
    charging_stations_df = map_data['charging_stations_df']
    delivery_points_df = map_data['delivery_points_df']
    
    charging_stations = set(charging_stations_df['pStationIntersection'].tolist())
    
    # Get delivery points by EV
    delivery_points_by_ev = {}
    for _, row in delivery_points_df.iterrows():
        ev_id = row['EV']
        delivery_point = row['pDeliveryIntersection']
        if ev_id not in delivery_points_by_ev:
            delivery_points_by_ev[ev_id] = set()
        delivery_points_by_ev[ev_id].add(delivery_point)
    
    # All delivery points (for intersection identification)
    all_delivery_points = set()
    for points in delivery_points_by_ev.values():
        all_delivery_points.update(points)
    
    # Get valid EVs (with solution data)
    valid_evs = [ev_id for ev_id, ev_results in all_ev_results.items() 
                if 'solution_data' in ev_results]
    
    # Create figure with N+1 subplots (N for each EV, 1 for aggregated demand)
    fig, axes = plt.subplots(len(valid_evs) + 1, 1, figsize=(16, 6 * (len(valid_evs) + 1)))
    
    # Create color mappings for EVs and charging stations
    ev_colors = plt.cm.tab10(np.linspace(0, 1, len(all_ev_results)))
    station_colors = plt.cm.Set3(np.linspace(0, 1, len(charging_stations)))
    
    # Create a mapping from charging station to color
    station_color_map = {station: station_colors[i] for i, station in enumerate(sorted(charging_stations))}
    
    # Determine time range (from starting_time to max_time)
    end_time = max_time
    
    # Plot individual EV trajectories
    for i, ev_id in enumerate(valid_evs):
        ax = axes[i]  # Get the appropriate subplot
        ev_results = all_ev_results[ev_id]
        
        # Get delivery points for current EV
        current_ev_delivery_points = delivery_points_by_ev.get(ev_id, set())
        
        if 'solution_data' not in ev_results:
            continue
            
        solution_data = ev_results['solution_data']
        intersections_df = solution_data['intersections_df']
        
        # Filter for visited intersections and sort by arrival time
        visited_intersections = intersections_df[
            abs(intersections_df['v01VisitIntersection'] - 1) < eps
        ].copy()
        visited_intersections = visited_intersections.sort_values('vTimeArrival')
        
        if len(visited_intersections) < 2:
            continue
            
        ev_color = ev_colors[i]
        
        # Create expanded trajectory points to split intersections where EV spends time
        expanded_trajectory = []
        
        # Create a mapping of intersections to their charging status for this EV
        charging_status_map = {}
        for _, row in visited_intersections.iterrows():
            intersection = row['intersection']
            is_charging = intersection in charging_stations and abs(row['v01Charge'] - 1) < eps
            charging_status_map[intersection] = is_charging
        
        for _, row in visited_intersections.iterrows():
            intersection = row['intersection']
            time_arr = row['vTimeArrival']
            time_dep = row['vTimeDeparture']
            soc_arr = row['vSoCArrival']
            soc_dep = row['vSoCDeparture']
            
            # Check if this is a location where the EV spends time
            is_charging = charging_status_map[intersection]
            is_delivery = intersection in current_ev_delivery_points
            
            # Always add arrival point
            arrival_point = {
                'intersection': intersection,
                'time': time_arr,
                'soc': soc_arr,
                'is_arrival': True,
                'is_departure': False,
                'is_charging': is_charging
            }
            expanded_trajectory.append(arrival_point)
            
            # If the EV spends time at this location, also add departure point as separate point
            if is_charging or is_delivery:
                # Only add departure point if there's a meaningful difference
                if abs(time_dep - time_arr) > eps or abs(soc_dep - soc_arr) > eps:
                    departure_point = {
                        'intersection': intersection,
                        'time': time_dep,
                        'soc': soc_dep,
                        'is_arrival': False,
                        'is_departure': True,
                        'is_charging': is_charging
                    }
                    expanded_trajectory.append(departure_point)
        
        # Sort expanded trajectory by time
        expanded_trajectory = sorted(expanded_trajectory, key=lambda x: x['time'])
        
        # Plot trajectory segments
        for j in range(len(expanded_trajectory) - 1):
            current_point = expanded_trajectory[j]
            next_point = expanded_trajectory[j + 1]
            
            # Determine segment color based on activity
            intersection = current_point['intersection']
            is_charging = current_point['is_charging']
            
            # Skip if this is a departure and the next point is not at a different intersection
            # This means it's the same intersection (arrival → departure)
            if (current_point['is_departure'] and not next_point['is_arrival'] and 
                current_point['intersection'] == next_point['intersection']):
                continue
                
            if intersection in charging_stations and is_charging and not current_point['is_departure']:
                # Use the same color as in the second plot for this charging station
                segment_color = station_color_map[intersection]
            elif intersection in current_ev_delivery_points and not current_point['is_departure']:
                segment_color = 'orange'  # delivery
            else:
                segment_color = 'black'  # default for travel
            
            # Plot segment
            times = [current_point['time'], next_point['time']]
            socs = [current_point['soc'], next_point['soc']]
            
            # Don't plot horizontal segments for charging/delivery (will be handled with special markers)
            if current_point['intersection'] == next_point['intersection']:
                # This is a stay at a single intersection (arrival → departure)
                # Don't plot this as a line segment, it will be marked with special symbols
                continue
            
            # Plot with thicker line for visibility
            ax.plot(times, socs, color=segment_color, linewidth=2.5, alpha=0.8)
        
        # Plot markers for all trajectory points
        for point in expanded_trajectory:
            intersection = point['intersection']
            time_point = point['time']
            soc_point = point['soc']
            is_arrival = point['is_arrival']
            is_departure = point['is_departure']
            is_charging = point['is_charging']
            
            # Determine marker style based on intersection type and arrival/departure
            if intersection in charging_stations and is_charging:
                # Only use charging station markers if EV actually charges here
                if is_arrival:
                    marker = '^'  # triangle up for arrival
                    marker_color = station_color_map[intersection]
                    marker_edge = 'darkblue'
                    marker_label = f"Arrival CS {int(intersection)}"
                else:
                    marker = 'v'  # triangle down for departure
                    marker_color = station_color_map[intersection]
                    marker_edge = 'darkblue'
                    marker_label = f"Departure CS {int(intersection)}"
            elif intersection in all_delivery_points and intersection in current_ev_delivery_points:
                # Only use delivery point markers if this is the current EV's delivery point
                if is_arrival:
                    marker = '^'  # triangle up for arrival
                    marker_color = 'orange'
                    marker_edge = 'darkorange'
                    marker_label = f"Arrival DP {int(intersection)}"
                else:
                    marker = 'v'  # triangle down for departure
                    marker_color = 'orange'
                    marker_edge = 'darkorange'
                    marker_label = f"Departure DP {int(intersection)}"
            else:
                # Use regular intersection marker for:
                # - Charging stations where EV doesn't charge
                # - Delivery points not belonging to current EV
                # - Regular intersections
                marker = 'o'  # circle for regular intersections
                marker_color = 'lightgray'
                marker_edge = 'gray'
                marker_label = f"Regular {int(intersection)}"
            
            # Plot marker - changed 'c=' to 'color=' to fix warning
            ax.scatter(time_point, soc_point, color=marker_color, marker=marker, 
                       s=150, edgecolors=marker_edge, linewidth=1, zorder=5)
            
            # Add intersection number label, showing arrival/departure status
            label_text = f"{int(intersection)}"
            
            # Set label color based on intersection type
            if intersection in charging_stations and is_charging:
                label_color = station_color_map[intersection]
            elif intersection in current_ev_delivery_points:
                label_color = 'orange'
            else:
                label_color = 'black'
                
            ax.annotate(label_text, (time_point, soc_point), 
                        xytext=(5, 5), textcoords='offset points',
                        fontsize=12, fontweight='bold', zorder=6, color=label_color)
        
        # Add a more prominent EV label at the start of each EV's journey
        first_point = expanded_trajectory[0]
        first_time = first_point['time']
        first_soc = first_point['soc']
        
        # Draw a prominent marker for each EV's starting point - changed 'c=' to 'color=' to fix warning
        ax.scatter(first_time, first_soc, color=ev_color, marker='*', 
                   s=300, edgecolors='black', linewidth=1, zorder=10)
        
        # Add vertical lines for charging and delivery activities
        for j in range(len(expanded_trajectory) - 1):
            current_point = expanded_trajectory[j]
            next_point = expanded_trajectory[j + 1]
            
            # Check if this is the same intersection (a stay at location)
            if current_point['intersection'] == next_point['intersection']:
                intersection = current_point['intersection']
                time_start = current_point['time']
                time_end = next_point['time']
                soc_start = current_point['soc']
                soc_end = next_point['soc']
                
                # Draw vertical line for the activity
                is_charging = current_point['is_charging']
                if intersection in charging_stations and is_charging:
                    # Vertical charging activity
                    ax.plot([time_start, time_end], [soc_start, soc_end], 
                            color=station_color_map[intersection], linewidth=3, 
                            linestyle='-', alpha=0.8, marker='')
                elif intersection in current_ev_delivery_points:
                    # Vertical delivery activity (mostly horizontal line as SoC doesn't change much)
                    ax.plot([time_start, time_end], [soc_start, soc_end], 
                            color='orange', linewidth=3, 
                            linestyle='-', alpha=0.8, marker='')
        
        # Format the EV plot
        ax.set_xlabel('Time (hours)', fontsize=12)
        ax.set_ylabel('State of Charge (%)', fontsize=12)
        ax.set_title(f'EV {ev_id} Trajectory: SoC vs Time', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Add time period vertical lines and labels
        for t in range(int(starting_time), int(end_time) + 1):  # starting_time to end_time to show all boundaries
            ax.axvline(x=t, color='gray', linestyle='--', alpha=0.5)
        
        # Add time period labels
        for t in range(int(starting_time), int(end_time)):
            hour = t % 24  # Convert to 24-hour format
            ax.text(t + 0.5, ax.get_ylim()[1] * 0.95, f'T{hour}\n({hour:02d}:00)', 
                    ha='center', va='top', fontsize=10, alpha=0.7)
        
        ax.set_xlim(starting_time - 0.5, end_time + 0.5)
        
        # Create legend elements for activity types and intersection types
        activity_legend = [
            plt.Line2D([0], [0], color='orange', linewidth=2, label='Delivery'),
            plt.Line2D([0], [0], color='black', linewidth=2, label='Travel'),
        ]
        
        # Add charging station colors to legend
        charging_legend = []
        for station in sorted(charging_stations):
            charging_legend.append(
                plt.Line2D([0], [0], color=station_color_map[station], linewidth=2, 
                          label=f'Charging at Station {int(station)}')
            )
        
        # Add marker types to legend
        marker_legend = [
            plt.Line2D([0], [0], marker='^', color='w', markerfacecolor='gray', 
                      markersize=10, label='Arrival'),
            plt.Line2D([0], [0], marker='v', color='w', markerfacecolor='gray', 
                      markersize=10, label='Departure'),
            plt.Line2D([0], [0], marker='*', color='w', markerfacecolor='red', 
                      markersize=13, label='EV Starting Point')
        ]
        
        # Add legends with increased fontsize
        # First legend for charging stations
        legend1 = ax.legend(handles=charging_legend, loc='upper right', title="Charging Activities", 
                            fontsize=11, bbox_to_anchor=(1.15, 1))
        ax.add_artist(legend1)
        
        # Second legend for activity types and markers
        legend2 = ax.legend(handles=activity_legend + marker_legend, loc='upper right', 
                            title="Activities & Markers", fontsize=11, bbox_to_anchor=(1.15, 0.7))
    
    # Plot aggregated demand as the last subplot
    ax = axes[len(valid_evs)]  # Get the last subplot
    
    # Calculate aggregated demand if not provided
    if aggregated_demand is None:
        # Pass the verbose parameter to extract_aggregated_demand
        aggregated_demand = extract_aggregated_demand(all_ev_results, map_data, eps, verbose)
    
    # Prepare data for stacked bar chart
    # Apply the starting time offset to the demand dataframe
    # Create a new dataframe with adjusted time periods
    demand_df_adjusted = pd.DataFrame()
    
    # Go through each time period in the original demand_df and map to actual hours
    for t in range(24):
        actual_hour = t % 24
        # Filter rows for current time period
        period_data = aggregated_demand[aggregated_demand['time_period'] == t].copy()
        period_data['actual_hour'] = actual_hour
        demand_df_adjusted = pd.concat([demand_df_adjusted, period_data], ignore_index=True)
    
    # Sort by actual hour for proper display
    demand_df_adjusted = demand_df_adjusted.sort_values(['charging_station', 'actual_hour']).reset_index(drop=True)
    
    # Create pivot table with actual hours for x-axis
    pivot_df = demand_df_adjusted.pivot(index='actual_hour', columns='charging_station', 
                                        values='aggregated_demand').fillna(0)
    
    # Sort index by hour for proper display
    pivot_df = pivot_df.sort_index()
    
    # Filter to only include hours from starting_time to max_time
    hours_to_include = list(range(int(starting_time), int(max_time)))
    pivot_df = pivot_df.loc[pivot_df.index.intersection(hours_to_include)]
    
    # Create stacked bar chart
    bottom = np.zeros(len(pivot_df))
    
    # Ensure we process charging stations in the same order for consistent colors
    for station in sorted(charging_stations):
        if station in pivot_df.columns:
            values = pivot_df[station].values
            ax.bar(pivot_df.index, values, bottom=bottom, 
                   label=f'Station {int(station)}', color=station_color_map[station], alpha=0.8)
            bottom += values
    
    # Format aggregated demand plot
    ax.set_xlabel('Time (hours)', fontsize=12)
    ax.set_ylabel('Aggregated Demand (kWh)', fontsize=12)
    ax.set_title('Aggregated Charging Demand by Time Period and Station', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add time period vertical lines
    for t in range(int(starting_time), int(max_time) + 1):
        ax.axvline(x=t - 0.5, color='gray', linestyle='--', alpha=0.5)
    
    # Adjust x-axis to match the actual hours starting from starting_time
    sorted_hours = sorted(pivot_df.index)
    ax.set_xticks(sorted_hours)
    
    # Create time period labels that match the actual hours
    time_labels = []
    for hour in sorted_hours:
        time_labels.append(f'T{hour}\n({hour:02d}:00-{hour:02d}:59)')
    
    ax.set_xticklabels(time_labels, rotation=45, ha='right', fontsize=10)
    
    # Set x-axis limits to match the actual hours
    ax.set_xlim(starting_time - 0.5, max_time - 0.5)
    
    # Add legend for aggregated demand plot with increased fontsize
    if len(pivot_df.columns) <= 10:  # Only show legend if not too many stations
        legend = ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(file_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    if verbose >= 1:
        print(f"Scenario analysis plots saved to: {file_path}")


