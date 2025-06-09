import gurobipy as gp


if __name__ == "__main__":

    model_path = "../gurobi_parameters/37-intersection map LIN EV1 Model.mps"
    parameters_prefix = "../gurobi_parameters/37-intersection map LIN EV1 Tuned Parameters"
    hours_tuning = 2

    params = {
        'TuneOutput': 2,
        'TuneTargetMIPGap': 0.05,
        'TuneTargetTime': 10,  # Try to reach target gap in less than this
        'TuneTimeLimit': 3600 * hours_tuning,
        'TuneTrials': 1,
        'TimeLimit': 60,  # Never solve for more than this
        #  'TuneJobs': 8,  # Run this number of threads in parallel --> Not allowed in the academic license...
    }

    model = gp.read(model_path)
    for param, val in params.items():
        model.setParam(param, val)
    
    print("Starting parameter tuning...")
    model.tune()
    
    # Save tuning results
    tune_result_count = model.getAttr('TuneResultCount')
    print(f"Found {tune_result_count} improved parameter sets.")
    
    if tune_result_count > 0:
        for i in range(tune_result_count):
            model.getTuneResult(i)
            result_file = f"{parameters_prefix} {i}.prm"
            model.write(result_file)
            print(f"Saved result {i} to {result_file}")
        print(f"Best parameters saved as {parameters_prefix} 0.prm")
    else:
        print("No improvements found.")

