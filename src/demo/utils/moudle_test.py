from utils.experiments import experiment_RQ4_a
models = ["bilstm","textcnn"]
features = ["tokens", "statistical"]
modes = ["WPDP","CPDP"]
datatypes = ["tokens","ast"]
for mode in modes:
    for datatype in datatypes:
        experiment_RQ4_a(mode,datatype)


