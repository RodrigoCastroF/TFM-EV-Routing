import gurobipy as gp


if __name__ == "__main__":

    params = {
        'TuneOutput': 3,
        'TuneTargetMIPGap': 0.05,
        'TuneTargetTime': 30,
        'TuneTimeLimit': 300,
        'TuneTrials': 1,
    }

    model_path = "../data/37-intersection map EV1 Model.mps"
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
            result_file = f"../data/tuned_params_{i}.prm"
            model.write(result_file)
            print(f"Saved result {i} to {result_file}")
        print(f"Best parameters saved as tuned_params_0.prm")
    else:
        print("No improvements found.")

