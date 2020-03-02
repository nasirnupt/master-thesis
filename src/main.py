import pickle
import argparse
import sys
from utils.experiments import experiment_RQ1, experiment_RQ2, experiment_RQ3,experiment_RQ4,experiment_RQ5

if __name__ == "__main__":
    modes = ["WPDP","CPDP"]
    datatypes = ["tokens","ast"]
    models = ["textcnn", "bilstm"]
    argment = sys.argv
    if argment[-1]=="RQ1":
        experiment_RQ1()
    elif argment[-1]=="RQ2":
        features = ["semantics","tokens","stats"]
        classifiers = ["lr","bilstm","textcnn"]
        for mode in modes:
            for classifier in classifiers:
                if classifier!="lr":

                    experiment_RQ2(mode=mode, feature="tokens", classifier=classifier)

                else:
                    for feature in features:
                        experiment_RQ2(mode=mode,feature=feature,classifier=classifier)


    elif argment[-1]=="RQ3":
        embeddings = ["word2vec","bert"]
        for mode in modes:
            for model in models:
                for embedding in embeddings:
                    experiment_RQ3(mode=mode,embedding=embedding,model=model)
    elif argment[-1]=="RQ4":
        for mode in modes:
            for datatype in datatypes:
                experiment_RQ4(mode, datatype)
    else:
        raise ValueError("String value is not in the 'RQ1','RQ2','RQ3' and 'RQ4'")










