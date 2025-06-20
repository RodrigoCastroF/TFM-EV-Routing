#!/usr/bin/env python3
"""
Script to create a map visualization of the 37-intersection network without solution data.
"""

from routing_model import load_excel_map_data, filter_map_data_for_ev, create_solution_map
import os


def main():
    """Generate a map image for the 37-intersection network without solution data."""
    
    # Configuration
    ev = 3  # This affects which delivery points are shown
    input_excel_file = "../data/37-intersection map.xlsx"
    output_image_file = f"../images/37-intersection map Custom EV{ev}.png"

    print(f"Loading data from {input_excel_file}...")
    map_data = load_excel_map_data(input_excel_file, verbose=1)

    print(f"Filtering data for EV {ev}...")
    input_data = filter_map_data_for_ev(map_data, ev)
    
    print(f"Creating map visualization without solution data...")
    create_solution_map(
        solution_data=None,
        input_data=input_data,
        file_path=output_image_file,
        ev=ev
    )
    print(f"Map image saved to: {output_image_file}")


if __name__ == "__main__":
    main() 