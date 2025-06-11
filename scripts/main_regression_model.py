"""
Main script to train profit regression models using the specified data files.
"""

from regression_model import train_profit_regression_model
import sys
import os
from datetime import datetime
from utils import TeeOutput


def main():
    # File paths
    scenarios_file = "../data/scenarios.csv"
    demand_file = "../data/training_data.csv"
    aggregator_excel_file = "../data/37-intersection map Aggregator Unrestricted.xlsx"
    output_folder = "../regressors"
    prefix = "37map_1001scenarios"
    
    # Train the regression models
    train_profit_regression_model(
        scenarios_file=scenarios_file,
        demand_file=demand_file,
        aggregator_excel_file=aggregator_excel_file,
        output_folder=output_folder,
        prefix=prefix,
        cv_folds=5,
        verbose=2
    )


if __name__ == "__main__":
    # Detailed logging
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file_path = f"../logs/regression_model_output_{timestamp}.txt"
    
    # Set up output redirection to both console and file
    if log_file_path:
        os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
    tee_output = TeeOutput(log_file_path)
    original_stdout = sys.stdout
    sys.stdout = tee_output
    
    try:
        print("=" * 60)
        now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print(f"Regression Model Training - {now}")
        print("=" * 60)
        print()
        
        main()
        
    except Exception as e:
        print(f"Error during execution: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Restore original stdout and close the log file
        sys.stdout = original_stdout
        tee_output.close()
        if log_file_path:
            print(f"All output has been saved to: {log_file_path}")
