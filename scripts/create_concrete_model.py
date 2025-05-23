from model import get_routing_map_data, get_ev_routing_abstract_model
from model.save_solution_data import save_solution_data
import pyomo.environ as pyo
from pyomo.opt import SolverFactory


def main(input_file, output_file, solver="gurobi", ev=1, time_limit=300, verbose=True):
    
    # Get data from Excel file
    print(f"Loading data from {input_file}...")
    input_data = get_routing_map_data(input_file, ev=ev)
    if verbose:
        print("Input data loaded successfully")
    
    # Get the abstract model
    print("Creating abstract model...")
    abstract_model = get_ev_routing_abstract_model()
    
    # Create a concrete instance using the data
    print("Creating concrete model instance...")
    concrete_model = abstract_model.create_instance(input_data)
    
    # Basic model information
    print("\nModel Information:")
    print(f"Number of intersections: {len(concrete_model.sIntersections)}")
    print(f"Number of paths: {len(concrete_model.sPaths)}")
    print(f"Number of delivery points: {len(concrete_model.sDeliveryPoints)}")
    print(f"Number of charging stations: {len(concrete_model.sChargingStations)}")
    
    # Create solver instance
    print(f"\nSetting up {solver} solver...")
    opt = SolverFactory(solver)
    
    # Set time limit based on solver
    time_limit_option = {"cbc": "seconds", "gurobi": "timeLimit", "glpk": "tmlim"}
    if solver in time_limit_option:
        opt.options[time_limit_option[solver]] = time_limit
        if verbose:
            print(f"Time limit set to {time_limit} seconds")
    
    # Solve the model
    print("Solving the model...")
    results = opt.solve(concrete_model, tee=verbose)
    
    if verbose:
        print("\nSOLVER RESULTS:")
        print(results)
    
    # Check if solution was found
    if (results.solver.status == pyo.SolverStatus.ok) and \
       (results.solver.termination_condition == pyo.TerminationCondition.optimal):
        print("\nOptimal solution found!")
        
        # Extract solution information
        try:
            lower_bound = results['problem'][0]['Lower bound']
            upper_bound = results['problem'][0]['Upper bound']
            if lower_bound is not None and lower_bound != 0:
                final_gap = (upper_bound - lower_bound) / abs(lower_bound)
            else:
                final_gap = 0.0
        except (KeyError, TypeError, IndexError):
            final_gap = 0.0
            lower_bound = None
            upper_bound = None
        
        try:
            execution_time = results['solver'][0]['Time']
        except (KeyError, IndexError):
            execution_time = None
            
        # Get objective function value
        obj_value = pyo.value(concrete_model.Obj)
        
        if verbose:
            print(f"Objective function value: {obj_value}")
            if final_gap is not None:
                print(f"Final gap: {final_gap:.6f}")
            if execution_time is not None:
                print(f"Execution time: {execution_time:.2f} seconds")
        
        # Save solution data to Excel
        print(f"\nSaving solution data to {output_file}...")
        try:
            save_solution_data(concrete_model, output_file)
            print("Solution data saved successfully!")
        except Exception as e:
            print(f"Error saving solution data: {e}")
        
        # Return results summary
        results_summary = {
            'model_instance': concrete_model,
            'solver_status': results.solver.status,
            'termination_condition': results.solver.termination_condition,
            'objective_value': obj_value,
            'final_gap': final_gap,
            'execution_time': execution_time,
            'lower_bound': lower_bound,
            'upper_bound': upper_bound
        }
        
        return results_summary
        
    elif results.solver.termination_condition == pyo.TerminationCondition.infeasible:
        print("\nModel is infeasible!")
        return {'status': 'infeasible', 'model_instance': concrete_model}
        
    elif results.solver.termination_condition == pyo.TerminationCondition.unbounded:
        print("\nModel is unbounded!")
        return {'status': 'unbounded', 'model_instance': concrete_model}
        
    else:
        print(f"\nSolver terminated with condition: {results.solver.termination_condition}")
        print("No optimal solution found.")
        return {'status': 'no_solution', 'model_instance': concrete_model, 'results': results}


if __name__ == "__main__":
    input_file = "../data/37-intersection map.xlsx"
    output_file = "../data/37-intersection map Solution.xlsx"
    results = main(input_file, output_file)
    print("Results:", results)

