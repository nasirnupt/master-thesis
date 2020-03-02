import os
import pandas as pd
import javalang
import string
from tqdm import tqdm
import numpy as np
from sklearn.metrics import f1_score
import pickle
import javalang
from javalang.tree import VariableDeclarator,MethodDeclaration,FormalParameter,BasicType,PackageDeclaration,InterfaceDeclaration
from javalang.tree import ClassDeclaration
from javalang.tree import SuperConstructorInvocation,MemberReference,SuperMemberReference
from javalang.tree import ConstructorDeclaration, ReferenceType,MethodInvocation,IfStatement, WhileStatement, DoStatement
from javalang.tree import ForStatement, AssertStatement,BreakStatement, ContinueStatement
from javalang.tree import ReturnStatement, SynchronizedStatement, TryStatement, SwitchStatement
from javalang.tree import BlockStatement, StatementExpression, TryResource, CatchClause
from javalang.tree import CatchClauseParameter, SwitchStatementCase, ForControl, EnhancedForControl
from sklearn.utils import shuffle


from utils.eda import *

defect_dir = "../data/code_repo/list"
src_dir = "../data/code_repo/data"
EMB_PATH = "../pretrained_models/word2vec/word2vec.vector"
complet_path = "/home/ljd/PycharmProjects/demo/data"


# data copy.
def gen_smote(x, alpha):
    columns = x.columns
    arg_data = pd.DataFrame(columns=columns)
    for column in columns:
        arg = []
        list_set = x[column].tolist()
        for i in range(len(x)):
            for j in range(alpha):
                arg.append(list_set[i])
        arg_data[column] = arg

    return arg_data

# generate more data with standard augmentation and imblance handdling.
def gen_eda(orig_x, orig_x1, orig_y, alpha, num_aug,datax_1=None):
    """
    Args:
        orig_x:sequence features,
        orig_x1:statistical features
        alpha: this determine how many word will be changed or replaced in a instance,
        num_aug:times of augmentation for each instance
    Return:
         sequence feautures, and DataFrame
    """
    data = pd.DataFrame(columns=["seq", "bug"])
    columns = orig_x1.columns
    data1 = pd.DataFrame(columns=list(orig_x1.columns))

    data_0 = []
    data_1 = []

    y_0 = []
    y_1 = []
    for i in range(len(orig_y)):
        if orig_y[i]==1:
            data_1.append(orig_x[i])
            y_1.append(orig_y[i])
        else:
            data_0.append(orig_x[i])
            y_0.append(orig_y[i])

    ir = len(data_0)//len(data_1)

    res = []
    for i in range(len(columns)):
        res.append([])
    arg_x = []
    arg_y = []
    if datax_1!=None:
        for i in range(len(data_1)):
            aug_sentences = eda(data_1[i], alpha_sr=alpha, alpha_ri=alpha, alpha_rs=alpha, p_rd=alpha, num_aug=ir)
            for sentence in aug_sentences:
                arg_x.append(sentence)
                arg_y.append(y_1[i])
                for j in range(len(res)):
                    res[j].append(datax_1[columns[j]].tolist()[i])

    for i in range(len(orig_x)):
        aug_sentences = eda(orig_x[i], alpha_sr=alpha, alpha_ri=alpha, alpha_rs=alpha, p_rd=alpha, num_aug=num_aug)
        for sentence in aug_sentences:
            arg_x.append(sentence)
            arg_y.append(orig_y[i])
            for j in range(len(res)):
                res[j].append(orig_x1[columns[j]].tolist()[i])

    for i in range(len(res)):
        data1[columns[i]] = res[i]

    data["seq"] = arg_x
    data["bug"] = arg_y
    data_sum = pd.concat([data,data1],axis=1)
    data_sum = data_sum.sample(frac=1.0)
    data_sum.reset_index(drop=True)
    data = data_sum[["seq","bug"]]
    data1 = data_sum[list(data1.columns)]
    del data_sum
    return data, data1

# cutting the name into two part: project name, version
def data_read(name):
    arr = name.split("-")
    print(arr)
    project_name = arr[0]
    version = arr[1][:-4]
    path = os.path.join(defect_dir,name)
    data = pd.read_csv(path)
    return data, project_name, version

# cut the number into several
def project_parsing(project,version,x):
    version_project = project+"-"+version
    file_path = os.path.join(src_dir, project)
    file_path = os.path.join(file_path, version_project)

    x = x.strip().split(".")
    x = "/".join(x)+".java"
    x = os.path.join(file_path, x)
    return x

def threshold_search(y_true, y_proba):
    best_threshold = 0
    best_score = 0

    for threshold in tqdm([i * 0.01 for i in range(100)]):
        score = f1_score(y_true=y_true, y_pred=y_proba > threshold)
        if score > best_score:
            best_threshold = threshold
            best_score = score
    search_result = {'threshold': best_threshold, 'f1': best_score}
    return search_result

# sequence generation: word tokens generation
def seq_gen(file):
    try:
        data = open(file).read()
        tokens = list(javalang.tokenizer.tokenize(data))
    except:
        return None

    res = []
    punc = list(string.punctuation)
    for token in tokens:
        if token.value not in punc:
            res.append(token.value)
    res = " ".join(res)

    # remove String
    res = re.sub('"*"',"",res)
    # filter non-alpha charater
    res = re.sub('[^a-zA-Z]+',' ',res)
    res = res.split(" ")
    count = 0
    while res[count]!="class" and count<len(res)-1:
        count += 1
    if count==len(res)-1:
        return None
    else:
        res = res[count:]
        return " ".join(res)

#this method are use AST node to generate sequence.
def ast_node_generation(file):
    class_list = [VariableDeclarator,MethodDeclaration,FormalParameter,BasicType,PackageDeclaration,InterfaceDeclaration,
                  CatchClauseParameter,ClassDeclaration,SuperConstructorInvocation,MemberReference,SuperMemberReference,
                  ConstructorDeclaration, ReferenceType,MethodInvocation,IfStatement, WhileStatement, DoStatement,
                  ForStatement, AssertStatement,BreakStatement, ContinueStatement,ReturnStatement, SynchronizedStatement,
                  TryStatement, SwitchStatement,BlockStatement, StatementExpression, TryResource, CatchClause,
                  SwitchStatementCase, ForControl, EnhancedForControl]
    class_collection = set()
    for item in class_list:
        class_collection.add(item)
    res = []
    try:
        string = open(file).read()
        tree = javalang.parse.parse(string)
    except:
        return None
    for _, node in tree:
        if type(node) in class_collection:
            try:
                res.append(node.name)
            except:
                continue
    return " ".join(res)

#embedding building
def get_coefs(word,*arr): return word, np.asarray(arr, dtype='float32')

def load_embeddings(embed_dir=EMB_PATH):
    embedding_index = dict(get_coefs(*o.strip().split(" ")) for o in tqdm(open(embed_dir)))
    return embedding_index

def build_embedding_matrix(word_index, embeddings_index, max_features, lower = True, verbose = True):
    embedding_matrix = np.zeros((max_features, 300))
    for word, i in tqdm(word_index.items(),disable = not verbose):
        if lower:
            word = word.lower()
        if i >= max_features: continue
        try:
            embedding_vector = embeddings_index[word]
        except:
            embedding_vector = embeddings_index["unknown"]
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector
    return embedding_matrix

def build_matrix(word_index, embeddings_index):
    embedding_matrix = np.zeros((len(word_index) + 1,300))
    for word, i in word_index.items():
        try:
            embedding_matrix[i] = embeddings_index[word]
        except:
            embedding_matrix[i] = embeddings_index["unknown"]
    return embedding_matrix


#embedding_matrix = build_matrix(word   _index, embeddings_index)

import os
non_list = []
drop_list = ['camel-1.0.csv', 'ivy-2.0.csv', 'poi-3.0.csv', 'lucene-2.0.csv', 'jedit-4.0.csv', 'synapse-1.0.csv']
result = {}

# solo data generation
def solo_data(path,mode="toks"):
    data, project_name, version = data_read(path)
    data["path"] = data["name.1"].apply(lambda x:project_parsing(project_name, version,x))
    if mode == "toks":
        data["seq"] = data["path"].apply(lambda x:seq_gen(x))
    else:
        data["seq"] = data["path"].apply(lambda x: ast_node_generation(x))
    data = data[data["seq"].notnull()]

    data_list = list(data.columns)
    data_list = [x for x in data_list if x not in ["name","version", "name.1","path","seq","bug"]]
    seq_feat = data["seq"]
    statics_feat = data[data_list]
    y = data["bug"].apply(lambda x:0 if x==0 else 1)
    return seq_feat, statics_feat, y

# merge data
def data_merge(name_list, dict):
    merge_seq_feat = None
    merge_stat_feat = None
    merge_y = None
    for item in name_list:
        seq_feat, stat_feat, y = dict[item]
        merge_seq_feat = pd.concat([merge_seq_feat,seq_feat])
        merge_stat_feat = pd.concat([merge_stat_feat, stat_feat])
        merge_y = pd.concat([merge_y, y])

    return merge_seq_feat, merge_stat_feat, merge_y


# generate the data that for projects data
def project_data_generation(mode="toks"):
    """
    the generated data of each project will be in such form:
    {project_name:
        {project_version1 : [seq_feat, stat_feat, y],
         project_version2 : [seq_feat, stat_feat, y],
         ......
         }
    the whole dataset will be in such form:
    [project1,project2,.....,projectn]
    """
    base_dir = "data/code_repo/list"
    dataset = {}
    camel_data_list = ["camel-1.0.csv", "camel-1.2.csv", "camel-1.4.csv", "camel-1.6.csv"]
    xalan_data_list = ["xalan-2.4.0.csv", "xalan-2.5.0.csv", "xalan-2.6.0.csv", "xalan-2.7.0.csv"]
    jedit_data_list = ["jedit-3.2.1.csv", "jedit-4.0.csv", "jedit-4.1.csv", "jedit-4.2.csv", "jedit-4.3.csv"]
    ant_data_list = ["ant-1.3.csv","ant-1.4.csv","ant-1.5.csv","ant-1.6.csv","ant-1.7.csv"]
    ivy_data_list = ["ivy-1.1.csv","ivy-1.4.csv","ivy-2.0.csv"]
    luncene_data_list = ["lucene-2.0.csv","lucene-2.2.csv","lucene-2.4.csv"]
    poi_data_list = ["poi-1.5.csv","poi-2.0RC1.csv","poi-2.5.1.csv","poi-3.0.csv"]
    synapse_data_list = ["synapse-1.0.csv","synapse-1.1.csv","synapse-1.2.csv"]
    velocity_data_list = ["velocity-1.4.csv","velocity-1.5.csv","velocity-1.6.1.csv"]
    xerces_data_list = ["xerces-init.csv","xerces-1.2.0.csv","xerces-1.3.0.csv","xerces-1.4.4.csv"]

    dataset["camel"] = camel_data_list
    dataset["xalan"] = xalan_data_list
    dataset["jedit"] = jedit_data_list
    dataset["ant"] = ant_data_list
    dataset["ivy"] = ivy_data_list
    dataset["luncene"] = luncene_data_list
    dataset["poi"] = poi_data_list
    dataset["synapse"] = synapse_data_list
    dataset["velocity"] = velocity_data_list
    dataset["xerces"] = xerces_data_list
    global defect_dir
    for k,v in dataset.items():
        temp = {}
        within_project = {}
        for item in v:
            seq_feat, stat_feat, y = solo_data(item,mode)
            print(seq_feat)
            if len(seq_feat)==0:
                continue
            within_project[item] = [seq_feat, stat_feat, y]
            #res.append(within_project)
        if len(within_project) == 0:
            continue
        dataset[k] = within_project
        temp[k] = within_project
        with open("../data/dataset_pkl/ast_node/"+k+".pkl","wb") as f:
            pickle.dump(temp,f)
    with open("../data/dataset_pkl/ast_node/dataset.pkl", "wb") as f:
        pickle.dump(dataset,f)

# data generation for WPDP
def WPDP_dataset_generation():
    """
    {project1:[[pre_project_version],[train_sequence_feature,train_statistical_features,train_y],
               [target_sequence_feature, target_statistical_feature, target_y]],
     project2:[[pre_project_version],[train_sequence_feature,train_statistical_features,train_y],
               [target_sequence_feature, target_statistical_feature, target_y]],
     ......
    }
    """
    data_dir = "../data/dataset_pkl/ast_node"
    wpdp_dataset = {}
    datalist = list(os.listdir(data_dir))
    datalist.remove("dataset.pkl")
    for item in datalist:
        path = os.path.join(data_dir, item)
        with open(path, "rb") as f:
            data_item = pickle.load(f)
        project_name = list(data_item.keys())[0]
        ver_data = data_item[project_name]

        # WPDP need at least 2 version project data
        if len(ver_data)<2:
            continue
        ver_name_list = list(ver_data.keys())
        pre_name = ver_name_list[-2]
        target_name = ver_name_list[-1]

        train_seq_feat, train_stat_feat, train_y = data_merge(ver_name_list[:-1], ver_data)
        target_seq_feat, target_stat_feat, target_y = ver_data[target_name][0],ver_data[target_name][1],ver_data[target_name][2]
        wpdp_dataset[target_name] = [[pre_name],[train_seq_feat,train_stat_feat,train_y],
                                     [target_seq_feat, target_stat_feat,target_y]]

    with open("../data/experiment_dataset/ast_node/wpdp_dataset.pkl", "wb") as f:
        pickle.dump(wpdp_dataset,f)

#data generation for CPDP

def CPDP_dataset_generation(mode="tokens"):
    """
        {target_project1:[[train_sequence_feature,train_statistical_features,train_y],
                   [target_sequence_feature, target_statistical_feature, target_y]],
         target_project2:[[train_sequence_feature,train_statistical_features,train_y],
                   [target_sequence_feature, target_statistical_feature, target_y]],
         ......

        }
        """
    data_dir = "../data/dataset_pkl/ast_node"
    cpdp_dataset = {}

    item_list = os.listdir(data_dir)
    item_list.remove("dataset.pkl")
    for i in range(len(item_list)):
        target_name = item_list[i]
        train_list = item_list[:i]+item_list[i+1:]
        with open(os.path.join(data_dir,target_name), "rb") as f:
            target_data = pickle.load(f)
        target_name = target_name.split(".")[0]
        target_data = target_data[target_name]
        target_data_list = list(target_data.keys())
        target_seq_feat, target_stat_feat, target_y = data_merge(target_data_list,target_data)

        merge_seq_feat = None
        merge_stat_feat = None
        merge_y = None
        for item in train_list:
            with open(os.path.join(data_dir, item), "rb") as f:
                train_data = pickle.load(f)
            item = item.split(".")[0]
            train_data = train_data[item]
            train_list = train_data.keys()
            train_seq_feat, train_stat_feat, train_y = data_merge(train_list, train_data)

            merge_seq_feat = pd.concat([merge_seq_feat, train_seq_feat])
            merge_stat_feat = pd.concat([merge_stat_feat, train_stat_feat])
            merge_y = pd.concat([merge_y, train_y])
        cpdp_dataset[target_name] = [[merge_seq_feat,merge_stat_feat,merge_y],
                                     [target_seq_feat, target_stat_feat, target_y]]
    with open("../data/experiment_dataset/ast_node/cpdp_dataset.pkl", "wb") as f:
        pickle.dump(cpdp_dataset, f)


# counting the range of token length

# this dataset for data generation.

#(1000,90%),(550,80%),(350,70%),(250,60%),(180,50%)--->tokens
#(200,90%),(120,80%),(80,70%),(58,60%),(44,50%)

if __name__ == "__main__":
    from utils.experiments import dataset_generation
    dataset, dataset_list = dataset_generation(mode="CPDP",datatype="ast")
    sample_name = dataset_list[0]
    train_seq_feat, target_seq_feat = dataset[sample_name][0][0], dataset[sample_name][1][0]
    dataset = pd.concat([train_seq_feat,target_seq_feat])
    dataset["seq_len"] = dataset.apply(lambda x:len(x.split()))
    import matplotlib.pyplot as plt
    import seaborn as sns
    leng = dataset["seq_len"].tolist()

    length = [x for x in leng if x <=43]
    print(len(length)/len(leng))
    sns.distplot(length)
    plt.show()
    #plt.savefig("../data/figure/distribution_of_length.png")
    # counting the percent of each coverage percent of distribution (90%, 80%, 70%, 60%, 50%)
















