
"""
from utils.experiments import dataset_generation
import matplotlib.pyplot as plt
import seaborn as sns

dataset, dataset_list = dataset_generation()
project_name = dataset_list[0]
target_seq_feat = dataset[project_name][1][0]
project_name = project_name.split("-")[0]
target_seq_feat["length"] = target_seq_feat.apply(lambda x:len(x.split(" ")))
sns.distplot(target_seq_feat["length"])
plt.xlabel("Length")
plt.ylabel("Frequency")
plt.savefig("../figures/WPDP_"+project_name+".png")
exit()
"""
#from utils.experiments import experiment_RQ4
#experiment_RQ4("WPDP")
#experiment_RQ4("WPDP","ast")
#exit()
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from utils.experiments import dataset_generation
dataset, dataset_list = dataset_generation("WPDP","ast")
res = {}

plt.figure(figsize=(12,10))
for i in range(len(dataset_list)):
    train_seq_feat = dataset[dataset_list[i]][1][0]
    filter_data = train_seq_feat.apply(lambda x:len(x.split()))
    if i>=8:

        plt.subplot(3,4,i+2)
    else:
        plt.subplot(3, 4, i + 1)

    sns.distplot(filter_data,hist=False,color="b",kde_kws={"shade":True})
    plt.xlabel(dataset_list[i].split("-")[0])
#plt.title("Length of distribution of each project in AST-node data type")
plt.show()





#data_list = filter_data.tolist()
#data_list.sort()
#res[project_name] = [data_list[len(data_list)//2],data_list[3*len(data_list)//5],data_list[7*len(data_list)//10],data_list[4*len(data_list)//5],data_list[9*len(data_list)//10]]


#df = pd.DataFrame(res)
#df.to_csv("../data/experiment_results/RQ1/WPDP_ast_coverage.csv",index=None)


#sns.distplot((dataset_550["length"]))
#plt.xlabel("Length")
#plt.ylabel("Frequency")
#plt.show()

from transformers import TFAlbertForSequenceClassification, BertForSequenceClassification
