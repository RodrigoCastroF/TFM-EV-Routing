import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import re
from typing import Dict, List, Tuple, Optional

def parse_gurobi_output_by_sections(content: str) -> Dict[Tuple[int, int], Tuple[List[float], List[float], List[float]]]:
    """
    Parse the output of the Gurobi solver by sections (scenario, EV pairs)
    to get the objective function values, gap values, and time stamps
    during the execution for each section.
    """
    
    # Position of relevant info in table (from the right)
    LOWER_BOUND_POS = 5
    GAP_POS = 3
    TIME_POS = 1
    
    max_pos = max(LOWER_BOUND_POS, GAP_POS, TIME_POS)
    
    # Dictionary to store data for each (scenario, EV) pair
    sections_data = {}
    
    # Split content by lines and look for section headers
    lines = content.split('\n')
    current_section = None
    current_data = []
    
    print("Parsing file content...")
    section_count = 0
    
    for i, line in enumerate(lines):
        # Look for section header with scenario and EV info
        if line.startswith('EV Routing Solver Output - Scenario'):
            # If we were processing a previous section, parse its data
            if current_section is not None and current_data:
                incumbent_values, gap_values, time_values = parse_section_data(current_data, max_pos)
                if incumbent_values:
                    sections_data[current_section] = (incumbent_values, gap_values, time_values)
                    print(f"  Found {len(incumbent_values)} data points for Scenario {current_section[0]}, EV {current_section[1]}")
            
            # Extract scenario and EV from header
            header_match = re.search(r'EV Routing Solver Output - Scenario (\d+), EV (\d+)', line)
            if header_match:
                scenario = int(header_match.group(1))
                ev = int(header_match.group(2))
                current_section = (scenario, ev)
                current_data = []
                section_count += 1
                print(f"Found section header: Scenario {scenario}, EV {ev}")
            else:
                print(f"Warning: Could not parse header: {line}")
                current_section = None
                current_data = []
        else:
            # Add line to current section data if we're in a section
            if current_section is not None:
                current_data.append(line)
    
    # Don't forget the last section
    if current_section is not None and current_data:
        incumbent_values, gap_values, time_values = parse_section_data(current_data, max_pos)
        if incumbent_values:
            sections_data[current_section] = (incumbent_values, gap_values, time_values)
            print(f"  Found {len(incumbent_values)} data points for Scenario {current_section[0]}, EV {current_section[1]}")
    
    print(f"Total sections found: {section_count}")
    print(f"Sections with data: {len(sections_data)}")
    
    return sections_data

def parse_section_data(section_lines: List[str], max_pos: int) -> Tuple[List[float], List[float], List[float]]:
    """
    Parse the data lines for a single section to extract incumbent, gap, and time values.
    """
    LOWER_BOUND_POS = 5
    GAP_POS = 3
    TIME_POS = 1
    
    incumbent_values = []
    gap_values = []
    time_values = []
    
    for line in section_lines:
        values = line.split()
        
        # Continue to next iteration if line does not have the desired info (GAP% and TIMEs)
        if len(values) < max_pos:
            continue
        if values[-GAP_POS].find("%") == -1 or values[-TIME_POS].find("s") == -1:
            continue
        
        try:
            incumbent_values.append(float(values[-LOWER_BOUND_POS]))
            gap_values.append(float(values[-GAP_POS].replace("%", "")))
            time_values.append(float(values[-TIME_POS].replace("s", "")))
        except (ValueError, IndexError):
            continue
    
    return incumbent_values, gap_values, time_values

def create_analysis_plot(input_file: str, output_file: str, 
                        scenarios_filter: Optional[List[int]] = None,
                        evs_filter: Optional[List[int]] = None):
    """
    Create a line graph showing lower bound vs time and gap vs time for multiple scenario-EV pairs.
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
    
    # Parse the Gurobi output by sections
    try:
        sections_data = parse_gurobi_output_by_sections(content)
    except Exception as e:
        print(f"Error parsing Gurobi output: {e}")
        return
    
    if not sections_data:
        print("No data found in the Gurobi output.")
        return
    
    # Filter data based on user preferences
    if scenarios_filter is not None or evs_filter is not None:
        filtered_data = {}
        for (scenario, ev), data in sections_data.items():
            include_scenario = scenarios_filter is None or scenario in scenarios_filter
            include_ev = evs_filter is None or ev in evs_filter
            if include_scenario and include_ev:
                filtered_data[(scenario, ev)] = data
        sections_data = filtered_data
    
    if not sections_data:
        print("No data found after applying filters.")
        return
    
    print(f"Found data for {len(sections_data)} scenario-EV pairs:")
    for (scenario, ev), (incumbent_values, gap_values, time_values) in sections_data.items():
        print(f"  Scenario {scenario}, EV {ev}: {len(incumbent_values)} data points")
        print(f"    Time range: {min(time_values):.2f}s - {max(time_values):.2f}s")
        print(f"    Lower bound range: {min(incumbent_values):.6f} - {max(incumbent_values):.6f}")
        print(f"    Gap range: {min(gap_values):.2f}% - {max(gap_values):.2f}%")
    
    # Create the plot
    fig, ax1 = plt.subplots(figsize=(14, 10))
    
    # Define colors for different scenario-EV pairs
    colors = plt.cm.tab10(np.linspace(0, 1, len(sections_data)))
    
    # Track all data for setting axis limits
    all_incumbent_values = []
    all_time_values = []
    all_gap_values = []
    
    # Plot data for each scenario-EV pair
    for i, ((scenario, ev), (incumbent_values, gap_values, time_values)) in enumerate(sections_data.items()):
        color = colors[i]
        
        # Convert to numpy arrays for easier manipulation
        time_array = np.array(time_values)
        incumbent_array = np.array(incumbent_values)
        gap_array = np.array(gap_values)
        
        # Filter out zero and negative values for log scale (time and incumbent only)
        valid_mask = (time_array > 0) & (incumbent_array > 0)
        time_filtered = time_array[valid_mask]
        incumbent_filtered = incumbent_array[valid_mask]
        gap_filtered = gap_array[valid_mask]
        
        if len(time_filtered) == 0:
            print(f"Warning: No valid data for Scenario {scenario}, EV {ev} after filtering for log scale")
            continue
        
        # Plot lower bound on primary y-axis
        label_incumbent = f'S{scenario}-EV{ev} Lower Bound'
        ax1.plot(time_filtered, incumbent_filtered, color=color, linewidth=2, 
                label=label_incumbent, linestyle='-', alpha=0.8)
        
        # Find the point with the best (lowest) lower bound for this section
        best_bound_idx = np.argmin(incumbent_filtered)
        best_bound_value = incumbent_filtered[best_bound_idx]
        best_bound_time = time_filtered[best_bound_idx]
        
        # Mark the best lower bound point
        ax1.plot(best_bound_time, best_bound_value, 'o', color=color, markersize=6, 
                markeredgecolor='black', markeredgewidth=1)
        
        # Add label next to the line (at the middle of the line)
        mid_idx = len(time_filtered) // 2
        ax1.annotate(f'S{scenario}-EV{ev}', 
                    xy=(time_filtered[mid_idx], incumbent_filtered[mid_idx]),
                    xytext=(5, 5), textcoords='offset points',
                    fontsize=9, color=color, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
        
        # Collect data for axis limits
        all_incumbent_values.extend(incumbent_filtered)
        all_time_values.extend(time_filtered)
        all_gap_values.extend(gap_filtered)
    
    # Set up primary y-axis (lower bound) with log scale
    ax1.set_xlabel('Execution Time (seconds)', fontsize=12)
    ax1.set_ylabel('Lower Bound (Incumbent)', color='tab:blue', fontsize=12)
    ax1.tick_params(axis='y', labelcolor='tab:blue')
    ax1.grid(True, alpha=0.3)
    
    # Set log scales
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    
    # Create second y-axis for gap
    ax2 = ax1.twinx()
    
    # Plot gap for each scenario-EV pair
    for i, ((scenario, ev), (incumbent_values, gap_values, time_values)) in enumerate(sections_data.items()):
        color = colors[i]
        
        # Convert to numpy arrays and filter for log scale (time and incumbent only)
        time_array = np.array(time_values)
        gap_array = np.array(gap_values)
        incumbent_array = np.array(incumbent_values)
        
        # Filter out zero and negative values for log scale (time and incumbent only)
        valid_mask = (time_array > 0) & (incumbent_array > 0)
        time_filtered = time_array[valid_mask]
        gap_filtered = gap_array[valid_mask]
        
        if len(time_filtered) == 0:
            continue
        
        label_gap = f'S{scenario}-EV{ev} Gap'
        ax2.plot(time_filtered, gap_filtered, color=color, linewidth=2, 
                label=label_gap, linestyle='--', alpha=0.6)
    
    ax2.set_ylabel('Gap (%)', color='tab:red', fontsize=12)
    ax2.tick_params(axis='y', labelcolor='tab:red')
    
    # Set axis limits with some padding
    if all_time_values and all_incumbent_values:
        ax1.set_xlim(min(all_time_values) * 0.9, max(all_time_values) * 1.1)
        ax1.set_ylim(min(all_incumbent_values) * 0.9, max(all_incumbent_values) * 1.1)
    
    if all_gap_values:
        ax2.set_ylim(min(all_gap_values) * 0.9, max(all_gap_values) * 1.1)
    
    # Add title
    filter_info = ""
    if scenarios_filter is not None:
        filter_info += f" (Scenarios: {scenarios_filter})"
    if evs_filter is not None:
        filter_info += f" (EVs: {evs_filter})"
    
    plt.title(f'Gurobi Solver Progress: Lower Bound and Gap vs Time{filter_info}', 
              fontsize=14, fontweight='bold')
    
    # Create legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    
    # Split legends into two columns if there are many entries
    if len(labels1) + len(labels2) > 6:
        legend1 = ax1.legend(lines1, labels1, loc='upper left', title='Lower Bound')
        legend2 = ax2.legend(lines2, labels2, loc='upper right', title='Gap (%)')
        legend1.get_title().set_fontweight('bold')
        legend2.get_title().set_fontweight('bold')
    else:
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='best')
    
    # Improve layout
    plt.tight_layout()
    
    # Save the plot
    try:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Plot saved successfully as: {output_file}")
    except Exception as e:
        print(f"Error saving plot: {e}")
        return
    
    # Optionally show the plot
    # plt.show()
    
    plt.close()

def main():
    """
    Main function to run the analysis.
    """
    
    # Define file paths
    input_file = "../logs/routing_solver_output_20250602_180046.txt"
    output_file = "../images/routing_solver_output_20250602_180046_analysis.png"
    
    # Filter options (None means show all)
    scenarios_filter = None  # e.g., [0, 1] to show only scenarios 0 and 1
    evs_filter = None        # e.g., [1, 2] to show only EVs 1 and 2
    
    print("Gurobi Output Analysis Script")
    print("=" * 50)
    print(f"Input file: {input_file}")
    print(f"Output file: {output_file}")
    print(f"Scenarios filter: {scenarios_filter if scenarios_filter else 'All'}")
    print(f"EVs filter: {evs_filter if evs_filter else 'All'}")
    print()
    
    # Create the analysis plot
    create_analysis_plot(input_file, output_file, scenarios_filter, evs_filter)

if __name__ == "__main__":
    main()
