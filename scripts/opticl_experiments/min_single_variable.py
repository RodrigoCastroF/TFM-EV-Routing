import opticl
import pandas as pd
import numpy as np
import os
from pyomo import environ
from pyomo.environ import *
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

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

# Train a regression model
version = 'single_var'
outcome = 'y'
alg = 'rf'  # Random Forest for non-linear relationship
task_type = 'continuous'
seed = 42

print(f"\nTraining {alg} model for outcome: {outcome}")

# Create results directory
if not os.path.exists(f'results/{alg}/'):
    os.makedirs(f'results/{alg}/')

# Train model
model_save = f'results/{alg}/{version}_{outcome}_model.csv'
m, perf = opticl.run_model(X_train, y_train, X_test, y_test, 'rf_shallow', outcome,
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

print(f"Model trained. R² score: {perf.iloc[0]['test_r2']:.4f}")

# Visualize the analytical function vs model predictions
print(f"\nCreating visualization of analytical vs predicted values...")

# Create a range of x values for plotting
x_plot = np.linspace(-3, 3, 300)
X_plot = pd.DataFrame(x_plot.reshape(-1, 1), columns=['x'])

# Calculate true analytical values
y_analytical = x_plot**2 + 2*x_plot

# Get model predictions
y_predicted = m.predict(X_plot)

# Create the plot
plt.figure(figsize=(12, 8))

# Plot the analytical function
plt.plot(x_plot, y_analytical, 'b-', linewidth=2, label='Analytical: y = x² + 2x', alpha=0.8)

# Plot the model predictions
plt.plot(x_plot, y_predicted, 'r--', linewidth=2, label='Random Forest Prediction', alpha=0.8)

# Plot training data points
plt.scatter(X_train['x'], y_train, alpha=0.3, s=20, c='gray', label='Training Data')

# Plot test data points
plt.scatter(X_test['x'], y_test, alpha=0.5, s=20, c='orange', label='Test Data')

# Mark the analytical minimum
plt.plot(-1, -1, 'go', markersize=10, label='Analytical Minimum (x=-1, y=-1)')

plt.xlabel('x')
plt.ylabel('y')
plt.title('Analytical Function vs Random Forest Prediction\ny = x² + 2x')
plt.legend()
plt.grid(True, alpha=0.3)
plt.xlim(-3, 3)

# Save the plot
plt.tight_layout()
plt.savefig(f'results/{alg}/{version}_{outcome}_function_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

# Calculate and display prediction accuracy statistics
mse = np.mean((y_analytical - y_predicted)**2)
mae = np.mean(np.abs(y_analytical - y_predicted))
max_error = np.max(np.abs(y_analytical - y_predicted))

print(f"\nPrediction accuracy on plotting range [-3, 3]:")
print(f"Mean Squared Error: {mse:.6f}")
print(f"Mean Absolute Error: {mae:.6f}")
print(f"Maximum Absolute Error: {max_error:.6f}")

# Set up optimization model
# Create model_master dataframe for opticl
performance = pd.DataFrame([{
    'alg': alg,  # This gets renamed to 'model_type' by model_selection
    'outcome': outcome,
    'valid_score': perf.iloc[0]['valid_score'],
    'save_path': model_save,
    'task': task_type,
    'seed': seed
}])

model_master = opticl.model_selection(performance, 
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

print(f"\nOptimization Results:")
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
else:
    print("Optimization failed!")
    print(f"Solver status: {results.solver.status}")

