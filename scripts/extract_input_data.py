import pandas as pd
from pathlib import Path


def extract_and_print_excel_data(excel_file_path):
    """
    Load an Excel file and print all sheets with their names and content.
    
    Parameters
    ----------
    excel_file_path : str
        Path to the Excel file
    """
    
    excel_file = Path(excel_file_path)
    
    if not excel_file.exists():
        print(f"Error: Excel file not found at {excel_file_path}")
        return
    
    try:
        # Load all sheets from the Excel file
        excel_data = pd.read_excel(excel_file_path, sheet_name=None)
        
        print(f"Excel file: {excel_file_path}")
        print(f"Number of sheets: {len(excel_data)}")
        print("=" * 80)
        
        # Iterate through each sheet
        for sheet_name, sheet_data in excel_data.items():
            print(f"\nSheet: '{sheet_name}'")
            print("-" * 60)
            print(f"Shape: {sheet_data.shape} (rows x columns)")
            print(f"Columns: {list(sheet_data.columns)}")
            print("\nContent:")
            print(sheet_data.to_string(index=False))
            print("=" * 80)
            
    except Exception as e:
        print(f"Error reading Excel file: {e}")


if __name__ == "__main__":
    # Path to the Excel file
    excel_file_path = "../data/37-intersection map.xlsx"
    
    # Extract and print all data
    extract_and_print_excel_data(excel_file_path) 