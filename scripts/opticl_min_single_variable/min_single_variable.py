import opticl
import pandas as pd
import numpy as np
import os
from pyomo import environ
from pyomo.environ import *
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import itertools

# Generate synthetic data with the specified relationship y = x^2 + 2x
np.random.seed(42)
n_samples = 1000

# Generate x values in a wider range to train the model properly
x_data = np.random.uniform(-3, 3, n_samples)
# Create the non-linear relationship y = x^2 + 2x with some noise
y_data = x_data**2 + 2*x_data + np.random.normal(0, 0.01, n_samples)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    x_data.reshape(-1, 1), y_data, test_size=0.2, random_state=42
)

# Convert to DataFrame format expected by opticl
X_train = pd.DataFrame(X_train, columns=['x'])
X_test = pd.DataFrame(X_test, columns=['x'])
y_train = pd.Series(y_train, name='y')
y_test = pd.Series(y_test, name='y')

print("Data generated:")
print(f"Training samples: {len(X_train)}")
print(f"Test samples: {len(X_test)}")
print(f"X range: [{X_train['x'].min():.2f}, {X_train['x'].max():.2f}]")
print(f"Y range: [{y_train.min():.2f}, {y_train.max():.2f}]")

# Train multiple regression models for comparison
version = 'single_var'
outcome = 'y'
alg_list = ['linear', 'rf', 'svm', 'cart', 'gbm', 'mlp']
task_type = 'continuous'
seed = 42

print(f"\nTraining multiple models for outcome: {outcome}")
print(f"Algorithms to test: {alg_list}")

# Store all models and performance data
all_models = {}
all_performance = []

# Train models for each algorithm
for alg in alg_list:
    print(f"\nTraining {alg} model...")
    
    # Create results directory
    if not os.path.exists(f'results/{alg}/'):
        os.makedirs(f'results/{alg}/')
    
    # Map algorithm names to opticl names
    alg_run = 'rf_shallow' if alg == 'rf' else alg
    
    try:
        # Train model
        model_save = f'results/{alg}/{version}_{outcome}_model.csv'
        m, perf = opticl.run_model(X_train, y_train, X_test, y_test, alg_run, outcome,
                                   task=task_type, 
                                   seed=seed, 
                                   cv_folds=5, 
                                   parameter_grid=None,
                                   save_path=model_save,
                                   save=False)
        
        # Save model for ConstraintLearning class
        constraintL = opticl.ConstraintLearning(X_train, pd.DataFrame(y_train), m, alg)
        constraint_add = constraintL.constraint_extrapolation(task_type)
        constraint_add.to_csv(model_save, index=False)
        
        # Save performance
        perf['seed'] = seed
        perf['outcome'] = outcome
        perf['alg'] = alg
        perf.to_csv(f'results/{alg}/{version}_{outcome}_performance.csv', index=False)
        
        # Store for comparison
        all_models[alg] = m
        all_performance.append(perf)
        
        print(f"{alg} - R² score: {perf.iloc[0]['test_r2']:.4f}")
        
    except Exception as e:
        print(f"Failed to train {alg}: {str(e)}")
        continue

# Create comprehensive performance comparison
if all_performance:
    performance_df = pd.concat(all_performance, ignore_index=True)
    performance_df.to_csv(f'results/{version}_performance_comparison.csv', index=False)
    
    # Display performance comparison
    print(f"\n{'='*60}")
    print("PERFORMANCE COMPARISON")
    print(f"{'='*60}")
    print(f"{'Algorithm':<10} {'R² Score':<12} {'Valid Score':<12} {'Test Score':<12}")
    print(f"{'-'*60}")
    
    for _, row in performance_df.iterrows():
        print(f"{row['alg']:<10} {row['test_r2']:<12.4f} {row['valid_score']:<12.4f} {row['test_score']:<12.4f}")
    
    # Find best performing algorithm based on test R² score
    best_alg_row = performance_df.loc[performance_df['test_r2'].idxmax()]
    best_alg = best_alg_row['alg']
    best_model = all_models[best_alg]
    
    print(f"\nBest performing algorithm: {best_alg} (R² = {best_alg_row['test_r2']:.4f})")
    
    # Visualize the analytical function vs predictions from all models
    print(f"\nCreating comprehensive visualization...")
    
    # Create a range of x values for plotting
    x_plot = np.linspace(-3, 3, 300)
    X_plot = pd.DataFrame(x_plot.reshape(-1, 1), columns=['x'])
    
    # Calculate true analytical values
    y_analytical = x_plot**2 + 2*x_plot
    
    # Create subplots for comparison
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    colors = ['red', 'green', 'blue', 'orange', 'purple', 'brown']
    
    for idx, (alg, model) in enumerate(all_models.items()):
        if idx < len(axes):
            ax = axes[idx]
            
            # Get model predictions
            y_predicted = model.predict(X_plot)
            
            # Plot the analytical function
            ax.plot(x_plot, y_analytical, 'b-', linewidth=2, label='Analytical: y = x² + 2x', alpha=0.8)
            
            # Plot the model predictions
            ax.plot(x_plot, y_predicted, '--', color=colors[idx], linewidth=2, 
                   label=f'{alg.upper()} Prediction', alpha=0.8)
            
            # Plot training data points (subset for clarity)
            sample_indices = np.random.choice(len(X_train), 50, replace=False)
            ax.scatter(X_train.iloc[sample_indices]['x'], y_train.iloc[sample_indices], 
                      alpha=0.3, s=15, c='gray')
            
            # Mark the analytical minimum
            ax.plot(-1, -1, 'go', markersize=8, label='Analytical Min')
            
            # Calculate accuracy metrics
            mse = np.mean((y_analytical - y_predicted)**2)
            mae = np.mean(np.abs(y_analytical - y_predicted))
            r2 = performance_df[performance_df['alg'] == alg]['test_r2'].iloc[0]
            
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_title(f'{alg.upper()}\nR² = {r2:.4f}, MSE = {mse:.4f}')
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
            ax.set_xlim(-3, 3)
    
    # Remove empty subplots
    for idx in range(len(all_models), len(axes)):
        fig.delaxes(axes[idx])
    
    plt.tight_layout()
    plt.savefig(f'results/{version}_all_algorithms_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Create detailed plot for best algorithm
    print(f"\nCreating detailed visualization for best algorithm ({best_alg})...")
    
    plt.figure(figsize=(12, 8))
    
    # Get best model predictions
    y_predicted_best = best_model.predict(X_plot)
    
    # Plot the analytical function
    plt.plot(x_plot, y_analytical, 'b-', linewidth=3, label='Analytical: y = x² + 2x', alpha=0.9)
    
    # Plot the best model predictions
    plt.plot(x_plot, y_predicted_best, 'r--', linewidth=3, 
            label=f'Best Model ({best_alg.upper()}) Prediction', alpha=0.9)
    
    # Plot training data points
    plt.scatter(X_train['x'], y_train, alpha=0.3, s=20, c='gray', label='Training Data')
    
    # Plot test data points
    plt.scatter(X_test['x'], y_test, alpha=0.5, s=20, c='orange', label='Test Data')
    
    # Mark the analytical minimum
    plt.plot(-1, -1, 'go', markersize=12, label='Analytical Minimum (x=-1, y=-1)')
    
    # Calculate and display prediction accuracy statistics
    mse_best = np.mean((y_analytical - y_predicted_best)**2)
    mae_best = np.mean(np.abs(y_analytical - y_predicted_best))
    max_error_best = np.max(np.abs(y_analytical - y_predicted_best))
    
    plt.xlabel('x', fontsize=12)
    plt.ylabel('y', fontsize=12)
    plt.title(f'Best Algorithm ({best_alg.upper()}) vs Analytical Function\n' + 
              f'R² = {best_alg_row["test_r2"]:.4f}, MSE = {mse_best:.6f}, MAE = {mae_best:.6f}', 
              fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.xlim(-3, 3)
    
    # Save the plot
    plt.tight_layout()
    plt.savefig(f'results/{version}_best_algorithm_{best_alg}.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"\nPrediction accuracy for best algorithm ({best_alg}) on plotting range [-3, 3]:")
    print(f"Mean Squared Error: {mse_best:.6f}")
    print(f"Mean Absolute Error: {mae_best:.6f}")
    print(f"Maximum Absolute Error: {max_error_best:.6f}")
    
    # Set up optimization model using the best performing algorithm
    print(f"\nSetting up optimization with best algorithm: {best_alg}")
    
    # Create model_master dataframe for opticl using best algorithm
    best_perf = performance_df[performance_df['alg'] == best_alg]
    performance_for_selection = pd.DataFrame([{
        'alg': best_alg,
        'outcome': outcome,
        'valid_score': best_perf.iloc[0]['valid_score'],
        'save_path': f'results/{best_alg}/{version}_{outcome}_model.csv',
        'task': task_type,
        'seed': seed
    }])
    
    model_master = opticl.model_selection(performance_for_selection, 
                                          constraints_embed=[], 
                                          objectives_embed={'y': 1})  # Minimize y
    
    # Add required columns
    model_master['lb'] = None
    model_master['ub'] = None  
    model_master['SCM_counterfactuals'] = None
    model_master['features'] = [['x']] * len(model_master.index)
    
    print(f"\nModel master configuration:")
    print(model_master)
    
    # Initialize optimization model
    model_pyo = ConcreteModel()
    
    # Create decision variable x with bounds [-2, 2]
    model_pyo.x = Var(['x'], domain=Reals, bounds=(-2, 2))
    
    # Initialize objective (will be updated by opticl)
    model_pyo.OBJ = Objective(expr=0, sense=minimize)
    
    print(f"\nSetting up optimization model...")
    
    # Embed the learned model and create final optimization model
    final_model_pyo = opticl.optimization_MIP(model_pyo, model_pyo.x, model_master, X_train, tr=True)
    
    print(f"Optimization model created with {len(list(final_model_pyo.component_objects(Constraint)))} constraints")
    
    # Solve the optimization problem
    opt = SolverFactory('gurobi')
    results = opt.solve(final_model_pyo, tee=False)
    
    print(f"\n{'='*60}")
    print("OPTIMIZATION RESULTS")
    print(f"{'='*60}")
    print(f"Using best algorithm: {best_alg}")
    print(f"Status: {results.solver.status}")
    print(f"Termination condition: {results.solver.termination_condition}")
    
    if results.solver.termination_condition == TerminationCondition.optimal:
        print(f"Objective value (minimized y): {final_model_pyo.OBJ():.4f}")
        
        x_sol = getattr(final_model_pyo, 'x')
        x_value = x_sol['x'].value
        print(f"Optimal x value: {x_value:.4f}")
        
        # Calculate the true function value for comparison
        true_y = x_value**2 + 2*x_value
        print(f"True y value (x² + 2x): {true_y:.4f}")
        
        # The analytical minimum of y = x² + 2x is at x = -1, y = -1
        print(f"\nAnalytical solution:")
        print(f"Minimum at x = -1, y = -1")
        print(f"Difference from analytical: x_diff = {x_value - (-1):.4f}, y_diff = {final_model_pyo.OBJ() - (-1):.4f}")
        
        # Calculate prediction at optimal point
        X_optimal = pd.DataFrame([[x_value]], columns=['x'])
        y_predicted_optimal = best_model.predict(X_optimal)[0]
        print(f"Model prediction at optimal x: {y_predicted_optimal:.4f}")
        print(f"Difference from true function: {abs(y_predicted_optimal - true_y):.4f}")
        
    else:
        print("Optimization failed!")
        print(f"Solver status: {results.solver.status}")

else:
    print("No models were successfully trained!")

