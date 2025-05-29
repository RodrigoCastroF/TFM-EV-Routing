import gurobipy as gp


if __name__ == "__main__":

    params = {
        'TuneOutput': 3,
        'TuneTargetMIPGap': 0.05,
        'TuneTargetTime': 15,
        'TuneTimeLimit': 60,
        'TuneTrials': 1,
    }

    model_path = "../data/37-intersection map EV1 Model.mps"
    model = gp.read(model_path)
    for param, val in params.items():
        model.setParam(param, val)
    model.tune()

