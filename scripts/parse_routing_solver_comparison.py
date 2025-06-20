import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import re
from typing import Dict, List, Tuple, Optional
from datetime import datetime

def parse_solver_output_by_sections(content: str) -> Dict[Tuple[str, str, int, int], Tuple[List[float], List[float]]]:
    """
    Parse the output of different solvers (Gurobi Linear, Gurobi Quadratic, CPLEX Linear)
    by sections (solver, constraints, scenario, EV) to get the incumbent values and time stamps.
    """
    
    # Dictionary to store data for each (solver, constraints, scenario, EV) tuple
    sections_data = {}
    
    # Split content by lines and look for section headers
    lines = content.split('\n')
    current_main_section = None  # (solver, constraints, scenario)
    current_ev = None
    current_ev_data = []
    
    print("Parsing file content...")
    section_count = 0
    
    i = 0
    while i < len(lines):
        line = lines[i]
        
        # Look for main section header with solver, constraints, and scenario info
        if line.startswith('EV Routing Solver Output - Solver'):
            # Process any pending EV data first
            if current_main_section is not None and current_ev is not None and current_ev_data:
                incumbent_values, time_values = parse_section_data(current_ev_data, current_main_section[0])
                if incumbent_values:
                    section_key = (current_main_section[0], current_main_section[1], current_main_section[2], current_ev)
                    sections_data[section_key] = (incumbent_values, time_values)
                    print(f"  Found {len(incumbent_values)} data points for {current_main_section[0]} {current_main_section[1]} S{current_main_section[2]} EV{current_ev}")
            
            # Extract solver, constraints, and scenario from header
            header_match = re.search(r'EV Routing Solver Output - Solver (\w+), Constraints (\w+), Scenario (\d+)', line)
            if header_match:
                solver = header_match.group(1)
                constraints = header_match.group(2)
                scenario = int(header_match.group(3))
                current_main_section = (solver, constraints, scenario)
                current_ev = None
                current_ev_data = []
                section_count += 1
                print(f"Found main section header: Solver {solver}, Constraints {constraints}, Scenario {scenario}")
            else:
                print(f"Warning: Could not parse header: {line}")
                current_main_section = None
                current_ev = None
                current_ev_data = []
        
        elif line.startswith('Processing EV') and current_main_section is not None:
            # Process any pending EV data first
            if current_ev is not None and current_ev_data:
                incumbent_values, time_values = parse_section_data(current_ev_data, current_main_section[0])
                if incumbent_values:
                    section_key = (current_main_section[0], current_main_section[1], current_main_section[2], current_ev)
                    sections_data[section_key] = (incumbent_values, time_values)
                    print(f"  Found {len(incumbent_values)} data points for {current_main_section[0]} {current_main_section[1]} S{current_main_section[2]} EV{current_ev}")
            
            # Extract EV number and start collecting data for this EV
            ev_match = re.search(r'Processing EV (\d+)', line)
            if ev_match:
                current_ev = int(ev_match.group(1))
                current_ev_data = []
                print(f"  Starting EV {current_ev} data collection")
        
        else:
            # Add line to current EV data if we're processing an EV within a section
            if current_main_section is not None and current_ev is not None:
                current_ev_data.append(line)
        
        i += 1
    
    # Don't forget the last EV data
    if current_main_section is not None and current_ev is not None and current_ev_data:
        incumbent_values, time_values = parse_section_data(current_ev_data, current_main_section[0])
        if incumbent_values:
            section_key = (current_main_section[0], current_main_section[1], current_main_section[2], current_ev)
            sections_data[section_key] = (incumbent_values, time_values)
            print(f"  Found {len(incumbent_values)} data points for {current_main_section[0]} {current_main_section[1]} S{current_main_section[2]} EV{current_ev}")
    
    print(f"Total main sections found: {section_count}")
    print(f"EV sections with data: {len(sections_data)}")
    
    return sections_data

def parse_section_data(section_lines: List[str], solver: str) -> Tuple[List[float], List[float]]:
    """
    Parse the data lines for a single EV section to extract incumbent and time values.
    """
    incumbent_values = []
    time_values = []
    
    if solver.lower() == 'gurobi':
        # Parse Gurobi output format
        in_optimization_table = False
        
        for line in section_lines:
            # Check if we're entering the optimization table
            if 'Nodes    |    Current Node    |     Objective Bounds' in line:
                in_optimization_table = True
                continue
            elif line.startswith('Cutting planes:') or line.startswith('Explored'):
                in_optimization_table = False
                continue
            
            if not in_optimization_table:
                continue
                
            # Look for lines with incumbent values in the optimization table
            # Pattern: H/*/nodes have incumbent, then 0.00000, then 100%, then time
            # Also regular node lines: nodes nodes status depth incumbent 0.00000 100% iter time
            if line.strip() and not line.startswith('-') and not line.startswith('|'):
                values = line.split()
                
                # Must end with time (ending with 's') and have enough columns
                if len(values) >= 5 and values[-1].endswith('s'):
                    try:
                        time_val = float(values[-1].replace('s', ''))
                        
                        # Find incumbent: look for the first decimal number > 0 that's not 0.00000
                        # and is followed by 0.00000 (best bound) and 100% (gap)
                        incumbent = None
                        
                        for i in range(len(values) - 3):  # Need space for incumbent, bestbound, gap
                            try:
                                val = float(values[i])
                                # Check if this looks like incumbent followed by bestbound and gap
                                if (val > 0 and '.' in values[i] and values[i] != '0.00000' and
                                    i + 2 < len(values) and 
                                    values[i + 1] == '0.00000' and 
                                    values[i + 2].endswith('%')):
                                    incumbent = val
                                    break
                            except ValueError:
                                continue
                        
                        if incumbent is not None:
                            incumbent_values.append(incumbent)
                            time_values.append(time_val)
                            print(f"Debug - Gurobi parsing: Found incumbent={incumbent}, time={time_val} from line: {line[:100]}")
                        
                    except (ValueError, IndexError):
                        continue
    
    elif solver.lower() == 'cplex':
        # Parse CPLEX output format
        elapsed_time = 0.0
        in_optimization_table = False
        
        for line in section_lines:
            # Look for elapsed time information
            if 'Elapsed time =' in line:
                time_match = re.search(r'Elapsed time = (\d+\.\d+) sec', line)
                if time_match:
                    elapsed_time = float(time_match.group(1))
                    continue
            
            # Check if we're entering the optimization table
            if 'Node  Left     Objective  IInf  Best Integer    Best Bound' in line:
                in_optimization_table = True
                continue
            elif line.strip().startswith('GUB cover cuts') or line.strip().startswith('Root node processing'):
                in_optimization_table = False
                continue
            
            # Look for lines with '*' (integer solution found) in CPLEX format
            if line.strip().startswith('*') and len(line.split()) >= 4:
                try:
                    # Extract objective value from CPLEX format
                    # CPLEX format variations: 
                    # *     0+    0                          247.8695        0.0000           100.00%
                    # * 10496+    0                          204.7155        0.0000           100.00%
                    
                    parts = line.split()
                    obj_val = None
                    
                    # Look for the objective value (typically a decimal number that's not 0.0000 or 100.00%)
                    for i, part in enumerate(parts[1:], 1):
                        try:
                            if '+' in part or part == '0' or part.startswith('0+'):
                                continue  # Skip node indicators
                            val = float(part)
                            # Skip percentages, small bounds, and obviously wrong values
                            if (val > 0.01 and val < 10000 and 
                                not part.endswith('%') and '.' in part and 
                                val != 0.0000):
                                obj_val = val
                                break
                        except ValueError:
                            continue
                    
                    if obj_val is not None:
                        incumbent_values.append(obj_val)
                        # Use elapsed time if available, otherwise estimate based on position
                        time_val = elapsed_time if elapsed_time > 0 else len(incumbent_values) * 0.5
                        time_values.append(time_val)
                        print(f"Debug - CPLEX parsing: Found incumbent={obj_val}, time={time_val} from line: {line[:80]}")
                    
                except (ValueError, IndexError):
                    continue
    
    # Clean up data - remove duplicates and sort by time
    if incumbent_values and time_values:
        # Combine and sort by time
        data_pairs = list(zip(time_values, incumbent_values))
        data_pairs.sort()  # Sort by time
        
        # Keep improving solutions and some intermediate points for better visualization
        print(f"Debug - Original data_pairs count: {len(data_pairs)}")
        
        if len(data_pairs) <= 1:
            # Not enough data to filter meaningfully
            if data_pairs:
                time_values, incumbent_values = zip(*data_pairs)
                return list(incumbent_values), list(time_values)
            else:
                return [], []
        
        # Sort by time and remove exact duplicates
        unique_pairs = []
        seen = set()
        for time_val, incumbent in data_pairs:
            pair_key = (time_val, incumbent)
            if pair_key not in seen:
                unique_pairs.append((time_val, incumbent))
                seen.add(pair_key)
        
        print(f"Debug - After removing duplicates: {len(unique_pairs)} points")
        
        # Keep improving solutions plus some intermediate points
        filtered_pairs = []
        best_so_far = float('inf')
        
        for time_val, incumbent in unique_pairs:
            # Keep if it's an improvement
            if incumbent < best_so_far:
                filtered_pairs.append((time_val, incumbent))
                best_so_far = incumbent
                print(f"Debug - Kept (improvement): time={time_val}, incumbent={incumbent}")
            # Also keep some intermediate points for time progression (every 10th point or every 2+ seconds)
            elif (len(filtered_pairs) > 0 and 
                  (len(filtered_pairs) % 10 == 0 or time_val - filtered_pairs[-1][0] >= 2.0)):
                filtered_pairs.append((time_val, incumbent))
                print(f"Debug - Kept (intermediate): time={time_val}, incumbent={incumbent}")
        
        print(f"Debug - Final filtered pairs: {len(filtered_pairs)} points")
        
        if filtered_pairs:
            time_values, incumbent_values = zip(*filtered_pairs)
            return list(incumbent_values), list(time_values)
    
    return [], []

def create_solver_comparison_plot(input_file: str, output_file: str):
    """
    Create a 3x3 grid plot showing incumbent values over time for each EV and scenario,
    with separate lines for each solver type.
    """
    
    # Read the input file
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            content = f.read()
    except FileNotFoundError:
        print(f"Error: File {input_file} not found.")
        return
    except Exception as e:
        print(f"Error reading file {input_file}: {e}")
        return
    
    # Parse the solver output by sections
    try:
        sections_data = parse_solver_output_by_sections(content)
    except Exception as e:
        print(f"Error parsing solver output: {e}")
        return
    
    if not sections_data:
        print("No data found in the solver output.")
        return
    
    # Organize data by scenario and EV
    scenarios = [0, 1, 2]
    evs = [1, 2, 3]
    solvers = ['gurobi_linear', 'gurobi_quadratic', 'cplex_linear']
    
    # Create organized data structure
    organized_data = {}
    for scenario in scenarios:
        organized_data[scenario] = {}
        for ev in evs:
            organized_data[scenario][ev] = {}
            for solver in solvers:
                organized_data[scenario][ev][solver] = {'incumbent': [], 'time': []}
    
    # Fill organized data structure
    for (solver, constraints, scenario, ev), (incumbent_values, time_values) in sections_data.items():
        solver_key = f"{solver}_{constraints}"
        if solver_key in solvers and scenario in scenarios and ev in evs:
            organized_data[scenario][ev][solver_key]['incumbent'] = incumbent_values
            organized_data[scenario][ev][solver_key]['time'] = time_values
    
    # Find the maximum time across all data to ensure consistent x-axis range
    max_time = 0
    for scenario in scenarios:
        for ev in evs:
            for solver in solvers:
                data = organized_data[scenario][ev][solver]
                if data['time']:
                    max_time = max(max_time, max(data['time']))
    
    print(f"Maximum time found across all data: {max_time:.2f} seconds")
    
    # Extend all solver lines to the maximum time with their final best value
    for scenario in scenarios:
        for ev in evs:
            for solver in solvers:
                data = organized_data[scenario][ev][solver]
                if data['incumbent'] and data['time']:
                    # If the last time point is not at max_time, extend the line
                    if data['time'][-1] < max_time - 0.01:  # Allow small tolerance for floating point comparison
                        # Add final point at max_time with the last (best) incumbent value
                        # Use the last incumbent value rather than the minimum to preserve the solution trajectory
                        final_incumbent = data['incumbent'][-1]
                        data['incumbent'].append(final_incumbent)
                        data['time'].append(max_time)
                        print(f"Extended {solver} S{scenario} EV{ev} to time {max_time:.2f} with incumbent {final_incumbent:.4f}")
                elif max_time > 0:
                    # If no data exists, create a placeholder line (this shouldn't happen with real data)
                    print(f"Warning: No data for {solver} S{scenario} EV{ev}, skipping extension")
    
    # Define colors for each solver
    colors = {
        'gurobi_linear': '#1f77b4',      # Blue
        'gurobi_quadratic': '#ff7f0e',   # Orange  
        'cplex_linear': '#2ca02c'        # Green
    }
    
    # Create the plot
    fig, axes = plt.subplots(3, 3, figsize=(18, 12))
    fig.suptitle(f'Solver Performance Comparison: Incumbent Values Over Time', fontsize=16, fontweight='bold')
    
    for i, scenario in enumerate(scenarios):
        for j, ev in enumerate(evs):
            ax = axes[i, j]
            
            # Plot data for each solver
            legend_added = False
            for solver in solvers:
                data = organized_data[scenario][ev][solver]
                if data['incumbent'] and data['time']:
                    incumbent_values = data['incumbent']
                    time_values = data['time']
                    
                    # Plot the line with step-post style to show that values are held constant
                    solver_name = solver.replace('_', ' ').replace('gurobi', 'Gurobi').replace('cplex', 'CPLEX').title()
                    ax.plot(time_values, incumbent_values, 
                           color=colors[solver], linewidth=2, 
                           label=solver_name, alpha=0.8, marker='o', markersize=3,
                           drawstyle='steps-post')  # Step plot shows incumbent is held until next improvement
                    legend_added = True
            
            # Set subplot properties
            ax.set_title(f'Scenario {scenario}, EV {ev}', fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3)
            
            # Set consistent x-axis limits across all subplots
            if max_time > 0:
                ax.set_xlim(0, max_time * 1.02)  # Add 2% padding
            
            # Use log scale if we have a wide range of values
            if legend_added:
                y_values = []
                for solver in solvers:
                    data = organized_data[scenario][ev][solver]
                    if data['incumbent']:
                        y_values.extend(data['incumbent'])
                
                # Check for log scale - avoid division by zero
                if y_values and len(y_values) > 1:
                    min_val = min(y_values)
                    max_val = max(y_values)
                    
                    # Only use log scale if min value is positive and there's a significant range
                    if min_val > 0.001 and max_val / min_val > 10:
                        ax.set_yscale('log')
                        print(f"Using log scale for Scenario {scenario}, EV {ev} (range: {min_val:.3f} to {max_val:.3f})")
                    elif min_val <= 0:
                        print(f"Linear scale for Scenario {scenario}, EV {ev} (contains non-positive values: min={min_val})")
            
            # Only show x-axis label for bottom row
            if i == 2:
                ax.set_xlabel('Time (seconds)', fontsize=11)
            
            # Only show y-axis label for leftmost column
            if j == 0:
                ax.set_ylabel('Incumbent Value', fontsize=11)
    
    # Create a single legend for the entire figure
    # Get legend from the first subplot that has data
    handles, labels = [], []
    for i in range(3):
        for j in range(3):
            h, l = axes[i, j].get_legend_handles_labels()
            if h:
                handles, labels = h, l
                break
        if handles:
            break
    
    if handles:
        fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.02), 
                  ncol=3, fontsize=14, frameon=True)
    
    # Adjust layout to prevent overlap
    plt.tight_layout()
    plt.subplots_adjust(top=0.93, bottom=0.12)
    
    # Save the plot
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {output_file}")
    
    # Show summary statistics
    print(f"\nSummary of data found (all lines extended to {max_time:.2f}s):")
    for scenario in scenarios:
        for ev in evs:
            print(f"Scenario {scenario}, EV {ev}:")
            for solver in solvers:
                data = organized_data[scenario][ev][solver]
                if data['incumbent']:
                    best_value = min(data['incumbent'])
                    time_range = f"{min(data['time']):.2f}s - {max(data['time']):.2f}s"
                    print(f"  {solver.replace('_', ' ').title()}: {len(data['incumbent'])} points, best = {best_value:.4f}, time range = {time_range}")
                else:
                    print(f"  {solver.replace('_', ' ').title()}: No data")

def main():
    """
    Main function to parse solver output and create analysis plot.
    """
    
    # Define input and output files
    input_file = "../logs/routing_solver_output_20250620_130731.txt"
    
    # Generate output filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"../images/routing_solver_comparison_{timestamp}.png"
    
    print(f"Parsing solver output from: {input_file}")
    print(f"Output plot will be saved to: {output_file}")
    
    # Create the analysis plot
    create_solver_comparison_plot(input_file, output_file)

if __name__ == "__main__":
    main()