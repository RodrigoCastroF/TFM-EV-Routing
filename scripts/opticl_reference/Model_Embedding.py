"""
This is an example of the OptiCL library created by its authors:
https://github.com/hwiberg/OptiCL/blob/main/notebooks/Pipeline/Model_Embedding.ipynb
"""

# %% [markdown]
# # Model Embedding Example

# %% [markdown]
# This script demonstrates how to implement a full model pipeline in which we have three outcomes of interest. We use our model training/selection procedure to fit a model for each outcome and embed these as objective terms and constraints.  a single model class, embed the model, and solve the optimization problem.

# %% [markdown]
# ## Load the relevant packages

# %%
import pandas as pd
import numpy as np
import math
from sklearn.utils.extmath import cartesian
import time
import sys
import os
import time
import itertools

# %%
import opticl
from pyomo import environ
from pyomo.environ import *

# %% [markdown]
# ## Initialize data
# We will work with a synthetic dataset using `sklearn` with three outcomes.

# %%
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
X, y = make_regression(n_samples=500, n_features = 10,
                       effective_rank = 5, n_targets = 3, 
                       noise = 5,
                       random_state=2)

## Add nonlinearities
y[:,1] = y[:,1]**2 
y[:,2] = np.log(y[:,2] - np.min(y[:,2]) + 1)

X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    random_state=1)
X_train = pd.DataFrame(X_train).add_prefix('col')
X_test = pd.DataFrame(X_test).add_prefix('col')

y_train = pd.DataFrame(y_train).add_prefix('y')
y_test = pd.DataFrame(y_test).add_prefix('y')

# %% [markdown]
# ## Train the chosen model type

# %% [markdown]
# We first train models for each outcome and algorithm pair. 
# 
# The user can optionally select a manual parameter grid for the cross-validation procedure. We implement a default parameter grid; see **run_MLmodels.py** for details on the tuned parameters. If you wish to use the default, leave ```parameter_grid = None``` (or do not specify any grid).
# 
# After training the model, we will save the trained model in the format needed for embedding the constraints. See **constraint_learning.py** for the specific format that is extracted per method. We also save the performance of the model to use in the automated model selection pipeline (if desired).
# 
# We also create the save directory if it does not exist.

# %%
version = 'test'
outcome_list = y_train.columns
# alg_list = ['linear','rf','svm','cart','gbm','mlp']
alg_list = ['rf']
task_type = 'continuous' # we are considering a regression problem

# %%
seed = 1

for outcome in outcome_list:
    print('Running models for outcome: %s' % outcome)
    for alg in alg_list:
        alg_run = 'rf_shallow' if alg == 'rf' else alg
        ## Train model
        model_save = 'results/%s/%s_%s_model.csv' % (alg, version, outcome)
        m, perf = opticl.run_model(X_train, y_train[outcome], X_test, y_test[outcome], alg_run, outcome,
                                   task = task_type, 
                                    seed = seed, 
                                    cv_folds = 5, 
                                    # The user can manually specify the parameter grid for cross-validation if desired (must match alg_run)
                                    parameter_grid = None,
                                    save_path = model_save,
                                    save = False)
        
        ## Save model for relevant ConstraintLearning class
        if not os.path.exists('results/%s/' % alg):
            os.makedirs('results/%s/' % alg)
        constraintL = opticl.ConstraintLearning(X_train, y_train, m, alg)
        constraint_add = constraintL.constraint_extrapolation(task_type)
        constraint_add.to_csv(model_save, index = False)

        ## Save performance
        perf['seed'] = seed
        perf['outcome'] = outcome
        perf['alg'] = alg
        perf.to_csv('results/%s/%s_%s_performance.csv' % (alg, version, outcome), index= False)

# %%
perf_files = ['results/%s/%s_%s_performance.csv' % (x[0], version, x[1]) 
              for x in itertools.product(alg_list, outcome_list)]
performance = pd.concat(pd.read_csv(x) for x in perf_files)
performance.to_csv('results/%s_performance.csv' % version, index = False)

# %% [markdown]
# ## Optimization formulation
# We will embed models for the outcomes using the methods trained above. The models will be selected using the model selection pipeline.
# 
# The key elements of the ``model_master`` dataframe are:
# - model_type: algorithm name.
# - outcome: name of outcome of interest; this is relevant in the case of multiple learned outcomes.
# - save_path: file name of the extracted model.
# - objective: the weight of the objective if it should be included as an additive term in the objective. A weight of 0 omits it from the objective entirely.
# - lb/ub: the lower (or upper) bound that we wish to apply to the learned outcome. If there is no bound, it should be set to ``None``.
# 
# In this case, we set the outcome 'y2' to be our objective term (with weight 1 to minimize; a weight of -1 would maximize the outcome), and 'y0' and 'y1' to be constraints.

# %%
model_master = opticl.model_selection(performance, 
                                      constraints_embed = ['y0','y1'], 
                                      objectives_embed = {'y2':1})

# %% [markdown]
# We also have to set an upper or lower bound for our constrained objectives, y0 and y1. For this example, we will constrain their upper bound to be the median in the training data. 'None' indicates no constraint.

# %%
y_medians = y_train.melt().groupby('variable').median()
model_master['lb'] = None
model_master['ub'] = None
model_master['SCM_counterfactuals'] = None
model_master['features'] = [[col for col in X_train.columns]]*len(model_master.index)
model_master.loc[model_master['outcome']=='y0', 'ub'] = y_medians.loc['y0','value']
model_master.loc[model_master['outcome']=='y1', 'ub'] = y_medians.loc['y1','value']
model_master

# %% [markdown]
# ### Initialize optimization model
# We begin by creating a base model (model_pyo) where we initialize our decision variables, fix any contextual variables, and specify domain-driven (known) constraints and objective terms.
# 
# For this synthetic example, we will fix the first two values of X (col0, col1) to the observed values in the first sample. In practice, sample would specify the contextual variables (w) and their known values that the user wants to optimize for.

# %%
sample = pd.DataFrame({'col0':[-.05],
                      'col1':[-.05]})

# %%
model_pyo = ConcreteModel()

## We will create our x decision variables.
N = X_train.columns
N_fixed = sample.columns
model_pyo.x = Var(N, domain=Reals)

## Fix the contextual features specified in 'sample'
def fix_value(model_pyo, index):
    return model_pyo.x[index] == sample.loc[0,index]

model_pyo.add_component('constr1_fixedvals', Constraint(N_fixed, rule=fix_value))

## Specify known constraints
model_pyo.add_component('constr_known1', Constraint(expr=sum(model_pyo.x[i] for i in N) <= 1))

## Specify any non-learned objective components - none here 
model_pyo.OBJ = Objective(expr=0, sense=minimize)

# %%
final_model_pyo = opticl.optimization_MIP(model_pyo, model_pyo.x, model_master, X_train, tr = True)
# final_model_pyo.pprint()

# %%
opt = SolverFactory('gurobi')
results = opt.solve(final_model_pyo) 

# %%
print("Objective value: %.3f" % final_model_pyo.OBJ())

print("\nX values: ")
x_sol = getattr(final_model_pyo, 'x')
for index in N:
    val = x_sol[index].value
    print("Feature %s: value = %.3f" % (index, val))
    
print("\nLambda values (convex hull weights): ")
lambda_sol = getattr(final_model_pyo, 'lam')
for index in lambda_sol:
    val = lambda_sol[index].value
    if val != 0:
        print("Observation %s: weight = %.3f" % (index, val))
        


