"""
Main script for solving the aggregator model with embedded OptiCL regression
"""

from aggregator_model import solve_aggregator_model
import sys
import os
from datetime import datetime
from utils import TeeOutput


def main(input_excel_file, performance_csv_file, training_data_csv_file, trust_region=True, output_excel_file=None,
         alg=None, solver="gurobi", time_limit=300, verbose=1, log_file=None):
    """
    Main function to solve the aggregator optimization problem with embedded regression.
    
    Parameters
    ----------
    input_excel_file: str
        Path to the input Excel file containing aggregator data.
    performance_csv_file: str
        Path to the CSV file containing performance comparison of regression models.
    training_data_csv_file: str
        Path to the CSV file containing training data for the regression models.
    trust_region: bool
        If True, solutions are restricted to lie in the trust region (the domain of the training data)
    output_excel_file: str, optional
        Path to save the solution Excel file (optional).
    alg: str, optional
        Algorithm to use for regression models (default: None for automatic selection).
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
            print(f"Aggregator Model Solver with OptiCL - {now}")
            print("=" * 60)
            print()

        # Solve the aggregator model
        results = solve_aggregator_model(
            input_excel_file=input_excel_file,
            performance_csv_file=performance_csv_file,
            training_data_csv_file=training_data_csv_file,
            trust_region=trust_region,
            output_excel_file=output_excel_file,
            alg=alg,
            solver=solver,
            time_limit=time_limit,
            verbose=verbose
        )

        if verbose >= 1:
            print("\nFinal Results:")
            print(f"  Solver status: {results.get('solver_status', 'unknown')}")
            if 'objective_value' in results:
                print(f"  Predicted profit: ${results['objective_value']:.2f}")
            if 'execution_time' in results and results['execution_time']:
                print(f"  Execution time: {results['execution_time']:.2f} seconds")
            
            if 'charging_prices' in results:
                print("  Optimal charging prices:")
                for station, price in results['charging_prices'].items():
                    if price is not None:
                        print(f"    Station {station}: ${price:.3f}/kWh")
                    else:
                        print(f"    Station {station}: UNKNOWN/NOT SET")
            
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
    trust_region = False
    model_type = 'competition'
    alg = 'rf'  # Algorithm to use (None for automatic selection, or specify: "linear", "rf", "svm", "cart", "gbm", "mlp")

    # Model type-specific files
    alg_suffix = f" {alg.title() if alg == 'linear' else alg.upper()}" if alg is not None else ""
    tr_suffix = " TR" if trust_region else ""
    
    model_type_files = {
        'restricted': (
            "../data/37-intersection map Aggregator Restricted.xlsx",
            "../regressors/37map_1001scenarios_performance_comparison.csv",
            "../regressors/37map_1001scenarios_training_data.csv",
            f"../solutions/37-intersection map Aggregator Restricted Solution{alg_suffix}{tr_suffix}.xlsx"
        ),
        'unrestricted': (
            "../data/37-intersection map Aggregator Unrestricted.xlsx",
            "../regressors/37map_1001scenarios_performance_comparison.csv",
            "../regressors/37map_1001scenarios_training_data.csv",
            f"../solutions/37-intersection map Aggregator Unrestricted Solution{alg_suffix}{tr_suffix}.xlsx"
        ),
        'competition': (
            "../data/37-intersection map Aggregator Competition.xlsx",
            "../regressors/37map_1001scenarios_competition_performance_comparison.csv",
            "../regressors/37map_1001scenarios_competition_training_data.csv",
            f"../solutions/37-intersection map Aggregator Competition Solution{alg_suffix}{tr_suffix}.xlsx"
        ),
    }[model_type]

    # Input files
    input_excel_file = model_type_files[0]
    performance_csv_file = model_type_files[1]
    training_data_csv_file =  model_type_files[2]
    
    # Output files
    output_excel_file =  model_type_files[3]

    # Detailed logging
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file_path = f"../logs/aggregator_solver_output_{timestamp}.txt"
    
    # Check if input files exist
    if not os.path.exists(input_excel_file):
        raise FileNotFoundError(f"Error: Input file not found: {input_excel_file}")
    if not os.path.exists(performance_csv_file):
        raise FileNotFoundError(f"Error: Performance CSV file not found: {performance_csv_file}")
    if not os.path.exists(training_data_csv_file):
        raise FileNotFoundError(f"Error: Training data CSV file not found: {training_data_csv_file}")
    
    # Create output directory if it doesn't exist
    if output_excel_file:
        os.makedirs(os.path.dirname(output_excel_file), exist_ok=True)
    if log_file_path:
        os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
    
    # Run the main function
    main(
        input_excel_file=input_excel_file,
        performance_csv_file=performance_csv_file,
        training_data_csv_file=training_data_csv_file,
        trust_region=trust_region,
        output_excel_file=output_excel_file,
        alg=alg,
        solver=solver,
        time_limit=time_limit,
        verbose=verbose,
        log_file=log_file_path
    )
