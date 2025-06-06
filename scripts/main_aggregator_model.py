"""
Main script for solving the aggregator model
"""

from aggregator_model import solve_aggregator_model
import sys
import os
from datetime import datetime


class TeeOutput:
    """Class to write output to both console and file simultaneously."""

    def __init__(self, file_path):
        self.terminal = sys.stdout
        self.log_file = None
        if file_path is not None:
            self.log_file = open(file_path, 'w', encoding='utf-8')

    def write(self, message):
        self.terminal.write(message)
        if self.log_file:
            self.log_file.write(message)
            self.log_file.flush()  # Ensure immediate writing to file

    def flush(self):
        self.terminal.flush()
        if self.log_file:
            self.log_file.flush()

    def close(self):
        if self.log_file:
            self.log_file.close()


def main(input_excel_file, output_excel_file=None, solver="gurobi", time_limit=300, verbose=1, log_file=None):
    """
    Main function to solve the aggregator optimization problem.
    
    Parameters
    ----------
    input_excel_file: str
        Path to the input Excel file containing aggregator data.
    output_excel_file: str, optional
        Path to save the solution Excel file (optional).
    solver: str
        Solver to use (default: "gurobi").
    time_limit: int
        Time limit in seconds (default: 300).
    verbose: int
        Verbosity level (0=silent, 1=basic, 2=detailed).
    log_file: str, optional
        Path to log file for saving output (optional).
    
    Returns
    -------
    dict
        Dictionary with solution results.
    """
    
    # Set up output redirection if log file specified
    original_stdout = None
    if log_file:
        original_stdout = sys.stdout
        sys.stdout = TeeOutput(log_file)
    
    try:
        if verbose >= 1:
            print("=" * 60)
            now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            print(f"Aggregator Model Solver - {now}")
            print("=" * 60)
            print()

        # Solve the aggregator model
        results = solve_aggregator_model(
            input_excel_file=input_excel_file,
            output_excel_file=output_excel_file,
            solver=solver,
            time_limit=time_limit,
            verbose=verbose
        )

        if verbose >= 1:
            print("\nFinal Results:")
            print(f"  Solver status: {results.get('solver_status', 'unknown')}")
            if 'objective_value' in results:
                print(f"  Total profit: ${results['objective_value']:.2f}")
            if 'execution_time' in results and results['execution_time']:
                print(f"  Execution time: {results['execution_time']:.2f} seconds")
            
            if 'charging_prices' in results:
                print("  Optimal charging prices:")
                for station, price in results['charging_prices'].items():
                    print(f"    Station {station}: ${price:.3f}/kWh")
            
            if output_excel_file:
                print(f"  Solution saved to: {output_excel_file}")
            if log_file:
                print(f"  Output logged to: {log_file}")
            print("=" * 60)

        return results

    finally:
        # Restore original stdout
        if original_stdout:
            if hasattr(sys.stdout, 'close'):
                sys.stdout.close()
            sys.stdout = original_stdout


if __name__ == "__main__":
    
    # Configuration
    solver = "gurobi"  # or "cbc", "glpk", "cplex"
    time_limit = 15  # seconds
    verbose = 2  # 0=silent, 1=basic, 2=detailed

    # Output files
    input_excel_file = "../data/37-intersection map Aggregator.xlsx"
    output_excel_file = "../data/37-intersection map Aggregator Solution.xlsx"

    # Detailed logging
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file_path = f"../data/aggregator_solver_output_{timestamp}.txt"
    
    # Check if input file exists
    if not os.path.exists(input_excel_file):
        raise FileNotFoundError(f"Error: Input file not found: {input_excel_file}")
    
    # Create output directory if it doesn't exist
    if output_excel_file:
        os.makedirs(os.path.dirname(output_excel_file), exist_ok=True)
    if log_file_path:
        os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
    
    # Run the main function
    main(
        input_excel_file=input_excel_file,
        output_excel_file=output_excel_file,
        solver=solver,
        time_limit=time_limit,
        verbose=verbose,
        log_file=log_file_path
    )
