"""
Main script to train profit regression models using the specified data files.
"""

from regression_model.train_profit_model import train_profit_regression_model


def main():
    # File paths
    scenarios_file = "../data/scenarios.csv"
    demand_file = "../data/training_data.csv"
    aggregator_excel_file = "../data/37-intersection map Aggregator.xlsx"
    output_folder = "../regressors"
    prefix = "profit_dummy"
    
    # Train the regression models
    train_profit_regression_model(
        scenarios_file=scenarios_file,
        demand_file=demand_file,
        aggregator_excel_file=aggregator_excel_file,
        output_folder=output_folder,
        prefix=prefix,
        cv_folds=2,
        verbose=2
    )


if __name__ == "__main__":
    main()
