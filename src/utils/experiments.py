from utils.preprocess import build_matrix,gen_eda
import numpy as np
import pandas as pd
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import f1_score
from models.model import textcnn
from models.model import bilstm_att_model,transformer_models,bert_lstm
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score
from transformers import TFAlbertModel, AlbertTokenizer
import keras_metrics as km
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import torch
early = EarlyStopping()
defect_dir = "./data/code_repo/list"
src_dir = "./data/code_repo/data"
EMB_PATH = "./pretrained_models/word2vec/word2vec.vector"
complet_path = "/home/ljd/PycharmProjects/demo/data"
import torch
from bert_pytorch.dataset import WordVocab


lr = 0.001
maxlen = 512
max_feature = 100000
embed_size = 300


non_list = []

"""
train_seq_feat: sequence features for training
target_seq_feat: sequence features for testing
train_stat_feat: statistical features for training
target_stat_feat: statistical features for testing
train_y: training label
target_y:testing label

maxlen: max sequence length for each instance
maxfeature: max vocabulary size
"""

# load top 'max_features' weight
def load_bert_weight(max_features):
    model = torch.load("./pretrained_models/bert/bert.model.ep1")
    tokenEmb = model.state_dict()["embedding.token.weight"]
    posEmb = model.state_dict()["embedding.position.pe"]
    return tokenEmb[:max_features,:].numpy(),posEmb[0].numpy()

# tokenize the text, if it > maxlen, cut the length==maxlen, else pad the length==maxlen,
# limit vocabulary size == max_features

def tokenize(text,maxlen,max_features,vocab):

    pad_index = vocab.pad_index
    unk_index = vocab.unk_index
    sequence = vocab.to_seq(text)
    length = len(sequence)
    for i in range(length):
        if sequence[i] > max_features:
            sequence[i] = unk_index
    if len(sequence) < maxlen:
        sequence += (maxlen-length)*[pad_index]
    else:
        sequence = sequence[:maxlen]
    return sequence
# bert embedding: position embedding and  token embedding
def bert_embedding(tokenEmb, posEmb,seq_data):
    res = []
    for i in range(len(seq_data)):
        res.append(np.array(tokenEmb[list(seq_data[i])])+np.array(posEmb))
    return np.array(res)

# this dataset for data generation. "ast node" or "full tokens"
def dataset_generation(mode="WPDP",datatype="tokens"):
    if mode=="WPDP":
        if datatype=="tokens":

            with open("./data/experiment_dataset/tokens/wpdp_dataset.pkl", "rb") as f:
                dataset = pickle.load(f)
            dataset_list = list(dataset.keys())
        else:

            with open("./data/experiment_dataset/ast_node/wpdp_dataset.pkl","rb") as f:
                dataset = pickle.load(f)
            dataset_list = list(dataset.keys())
    else:
        if datatype=="tokens":
            with open("./data/experiment_dataset/tokens/cpdp_dataset.pkl", "rb") as f:
                dataset = pickle.load(f)
            dataset_list = list(dataset.keys())
        else:
            with open("./data/experiment_dataset/ast_node/cpdp_dataset.pkl","rb") as f:
                dataset = pickle.load(f)
            dataset_list = list(dataset.keys())

    return dataset, dataset_list
# experiment for data augmentation on both WPDP and CPDP whether it if effective to our model
def embedding_matrix_generation(word_index):
    with open("./data/embedding_index.pkl", "rb") as f:
        embedding_index = pickle.load(f)
    embedding_matrix = build_matrix(word_index, embedding_index)
    return embedding_matrix

# oversampling to handle imblance problem.
def data_oversampling(train_seq_feat, train_stat_feat, train_y):
    train_stat_1 = pd.concat([train_stat_feat, train_y], axis=1)
    train_stat_1 = train_stat_1[train_stat_1["bug"] == 1]
    train_stat_1 = train_stat_1[list(train_stat_feat.columns)]

    train_seq_1 = pd.concat([train_seq_feat, train_y], axis=1)
    train_seq_1 = train_seq_1[train_seq_1["bug"] == 1]

    train_y_1 = train_seq_1["bug"]
    train_seq_1 = train_seq_1["seq"]

    train_stat_0 = pd.concat([train_stat_feat, train_y], axis=1)
    train_stat_0 = train_stat_0[train_stat_0["bug"] == 0]
    train_stat_0 = train_stat_0[list(train_stat_feat.columns)]

    train_seq_0 = pd.concat([train_seq_feat, train_y], axis=1)
    train_seq_0 = train_seq_0[train_seq_0["bug"] == 0]
    train_y_0 = train_seq_0["bug"]
    train_seq_0 = train_seq_0["seq"]

    if len(train_seq_0)<len(train_seq_1):
        ir =  len(train_seq_1) // len(train_seq_0)
        for _ in range(ir-1):
            train_seq_feat = pd.concat([train_seq_feat, train_seq_0])
            train_y = pd.concat([train_y, train_y_0])
            train_stat_feat = pd.concat([train_stat_feat, train_stat_0])
    else:
        ir = len(train_seq_0) // len(train_seq_1)
        for _ in range(ir-1):
            train_seq_feat = pd.concat([train_seq_feat, train_seq_1])
            train_y = pd.concat([train_y, train_y_1])
            train_stat_feat = pd.concat([train_stat_feat, train_stat_1])
    return train_seq_feat,train_stat_feat,train_y

# data augmentation.
#1. sequence generation
#2. imblance handling
#3. data gumentation for [1,2,3,4,5] times
def experiment_RQ1(mode="WPDP",datatype="tokens"):
    res = {}
    dataset, dataset_list = dataset_generation(mode)
    vocab = WordVocab.load_vocab("./pretrained_models/bert/vocab.txt")

    tokenEmb, posEmb = load_bert_weight(max_feature)
    count = 0
    for project_name in dataset_list:
        print(project_name)
        res_in = {}
        if len(dataset[project_name][1][0])>=1000:
            continue
        if count == 4:
            break
        count += 1
        for i in [0,2,4,8,16,32]:
            if mode == "WPDP":
                pre_project_name = dataset[project_name][0][0]
                train_seq_feat, train_stat_feat, train_y = dataset[project_name][1][0], dataset[project_name][1][1], \
                                                           dataset[project_name][1][2]
                target_seq_feat, target_stat_feat, target_y = dataset[project_name][2][0], dataset[project_name][2][1], \
                                                              dataset[project_name][2][2]
            else:
                train_seq_feat, train_stat_feat, train_y = dataset[project_name][0][0], dataset[project_name][0][1], \
                                                           dataset[project_name][0][2]
                target_seq_feat, target_stat_feat, target_y = dataset[project_name][1][0], dataset[project_name][1][1], \
                                                              dataset[project_name][1][2]

            #oversampling
            train_seq_feat, train_stat_feat, train_y = data_oversampling(train_seq_feat,train_stat_feat,train_y)
            # data generation: generated times: i
            new_data, train_stat_feat = gen_eda(train_seq_feat.tolist(),train_stat_feat, train_y.tolist(),0.1,i)
            #new_data, statics_feat = gen_eda(seq_feat.tolist(), statics_feat, y.tolist(), 0.1, i)
            train_seq_feat = new_data["seq"]
            train_y = new_data["bug"]
            maxlen = 512
            del new_data
            print("processing begin..")
            train_seq_feat = train_seq_feat.apply(lambda x: tokenize(x, 512, max_feature, vocab))
            target_seq_feat = target_seq_feat.apply(lambda x: tokenize(x, 512, max_feature, vocab))
            print("processing finished")
            train_seq_feat = np.array(list(train_seq_feat))
            target_seq_feat = np.array(list(target_seq_feat))
            train_posEmb = np.expand_dims(posEmb, 0).repeat(train_seq_feat.shape[0], axis=0)
            target_posEmbd = np.expand_dims(posEmb, 0).repeat(target_seq_feat.shape[0], axis=0)
            pred = bert_lstm(train_seq_feat, train_y, target_seq_feat, target_y,
                             tokenEmb, train_posEmb, target_posEmbd, max_feature)
            f1 = f1_score(target_y, pred)
            precision = precision_score(target_y, pred)
            recall = recall_score(target_y, pred)

            num_project_name = project_name + str(i)
            res_in[num_project_name] = [round(f1,2), round(precision,2),round(recall,2)]
        res[project_name] = res_in
    with open("./data/experiment_results/RQ1/"+mode+".pkl","wb") as f:
        pickle.dump(res,f)

# can our model beter than traditional model
# compare with statistical features
# 1.model1: statistical features + Logistics Regression
# 2.model2: tokenized sequence + Logistics Regression
# 3.BERT model: semantics features + Logistics Regression

def experiment_RQ2(mode="WPDP", feature="semantics",classifier="lr",datatype="tokens"):
    """
    Args:
        mode: WPDP or CPDP
        feature: "semantics","token" or "statistical"
        classifer: given only set 'model=traditional'
        classifier: logistics regression.
    """
    dataset, dataset_list = dataset_generation(mode,datatype)
    vocab = WordVocab.load_vocab("./pretrained_models/bert/vocab.txt")
    res={}
    tokenEmb, posEmb = load_bert_weight(max_feature)
    for project_name in dataset_list:
        if mode=="WPDP":

            pre_project_name = dataset[project_name][0][0]
            train_seq_feat, train_stat_feat, train_y = dataset[project_name][1][0], dataset[project_name][1][1], \
                                                       dataset[project_name][1][2]
            target_seq_feat, target_stat_feat, target_y = dataset[project_name][2][0], dataset[project_name][2][1], \
                                                          dataset[project_name][2][2]
        else:
            train_seq_feat, train_stat_feat, train_y = dataset[project_name][0][0], dataset[project_name][0][1], \
                                                       dataset[project_name][0][2]
            target_seq_feat, target_stat_feat, target_y = dataset[project_name][1][0], dataset[project_name][1][1], \
                                                          dataset[project_name][1][2]

        train_seq_feat, train_stat_feat, train_y = data_oversampling(train_seq_feat, train_stat_feat, train_y)
        if mode=="WPDP":
            new_data, train_stat_feat = gen_eda(train_seq_feat.tolist(), train_stat_feat, train_y.tolist(), 0.1, 3)
            # new_data, statics_feat = gen_eda(seq_feat.tolist(), statics_feat, y.tolist(), 0.1, i)
            train_seq_feat = new_data["seq"]
            train_y = new_data["bug"]
            del new_data
        if classifier=="lr":
            print("processing begin..")
            train_seq_feat = train_seq_feat.apply(lambda x: tokenize(x,512,max_feature,vocab))
            target_seq_feat = target_seq_feat.apply(lambda x: tokenize(x,512,max_feature,vocab))
            print("processing finished")
            train_seq_feat = np.array(list(train_seq_feat))
            target_seq_feat = np.array(list(target_seq_feat))
            if feature=="semantics":
                train_posEmb = np.expand_dims(posEmb, 0).repeat(train_seq_feat.shape[0], axis=0)
                target_posEmbd = np.expand_dims(posEmb, 0).repeat(target_seq_feat.shape[0], axis=0)
                pred = bert_lstm(train_seq_feat,train_y,target_seq_feat,target_y,
                                           tokenEmb,train_posEmb,target_posEmbd,max_feature)
            else:
                classification = LogisticRegression()
                if feature == "tokens":
                    classification.fit(train_seq_feat, train_y)
                    pred = classification.predict(target_seq_feat)
                else:
                    classification.fit(train_stat_feat, train_y)
                    pred = classification.predict(target_stat_feat)
            f1 = f1_score(target_y, pred)
            precision = precision_score(target_y,pred)
            recall = recall_score(target_y,pred)
        else:

            tokenizer = Tokenizer(num_words=max_feature, lower=False)
            tokenizer.fit_on_texts(list(train_seq_feat) + list(target_seq_feat))
            word_index = tokenizer.word_index
            train_seq_feat = tokenizer.texts_to_sequences(list(train_seq_feat))
            train_seq_feat = pad_sequences(train_seq_feat, maxlen=maxlen)

            target_seq_feat = tokenizer.texts_to_sequences(list(target_seq_feat))
            target_seq_feat = pad_sequences(target_seq_feat, maxlen=maxlen)
            # load the embedding index
            with open("./data/embedding_index.pkl", "rb") as f:
                embedding_index = pickle.load(f)
            embedding_matrix = build_matrix(word_index, embedding_index)

            if classifier == "textcnn":
                f1, precision, recall = textcnn("word2vec",train_seq_feat, train_y, word_index, target_seq_feat, target_y,
                                                embedding=embedding_matrix, maxlen=maxlen)
            else:
                f1, precision, recall = bilstm_att_model("word2vec",train_seq_feat, train_y, target_seq_feat, target_y, word_index,
                                                         64, 2, embedding=embedding_matrix)

        if mode=="WPDP":
            res[pre_project_name] = [round(f1,2),round(precision,2),round(recall,2)]
        else:
            res[project_name] = [round(f1,2),round(precision,2),round(recall,2)]
    df = pd.DataFrame(res)

    if classifier!="lr":
        df.to_csv("./data/experiment_results/RQ2/"+mode+"_"+classifier+".csv",index=False)
    else:
        df.to_csv("./data/experiment_results/RQ2/" + mode + "_" + feature + "_" + classifier + ".csv", index=False)
# compare with deep learning models.
# bert model VS textCNN and bilstm
def experiment_RQ2_B(mode="WPDP",datatype="ast", model="textcnn",):
    dataset, dataset_list = dataset_generation(mode=mode,datatype=datatype)
    res = {}
    for project_name in dataset_list:
        if mode == "WPDP":
            pre_project_name = dataset[project_name][0][0]
            train_seq_feat, train_stat_feat, train_y = dataset[project_name][1][0], dataset[project_name][1][1], \
                                                       dataset[project_name][1][2]
            target_seq_feat, target_stat_feat, target_y = dataset[project_name][2][0], dataset[project_name][2][1], \
                                                          dataset[project_name][2][2]
        else:
            train_seq_feat, train_stat_feat, train_y = dataset[project_name][0][0], dataset[project_name][0][1], \
                                                       dataset[project_name][0][2]
            target_seq_feat   , target_stat_feat, target_y = dataset[project_name][1][0], dataset[project_name][1][1], \
                                                          dataset[project_name][1][2]
        train_seq_feat, train_stat_feat, train_y = data_oversampling(train_seq_feat, train_stat_feat, train_y)
        #if augmentation:
        #    new_data, stat_feat = gen_eda(seq_feat.tolist(), stat_feat, y.tolist(), 0.1, argumentation)
        #    seq_feat = new_data["seq"]
        #    y = new_data["bug"]
        #    del new_data
        #    del new_data
        maxlen = 512
        if model!="bert":
            tokenizer = Tokenizer(num_words=max_feature, lower=False)
            tokenizer.fit_on_texts(list(train_seq_feat)+list(target_seq_feat))
            word_index = tokenizer.word_index
            train_seq_feat = tokenizer.texts_to_sequences(list(train_seq_feat))
            train_seq_feat = pad_sequences(train_seq_feat, maxlen=maxlen)

            target_seq_feat = tokenizer.texts_to_sequences(list(target_seq_feat))
            target_seq_feat = pad_sequences(target_seq_feat,maxlen=maxlen)
            #load the embedding index
            with open("./data/embedding_index.pkl", "rb") as f:
                embedding_index = pickle.load(f)
            embedding_matrix = build_matrix(word_index, embedding_index)

            if model=="textcnn":
                f1, precision, recall = textcnn(train_seq_feat,train_y,word_index,target_seq_feat,target_y,embedding=embedding_matrix,maxlen=maxlen)
            else:
                f1, precision, recall =bilstm_att_model(train_seq_feat,train_y,target_seq_feat,target_y,word_index,64,2,embedding=embedding_matrix)
        else:
            from transformers import TFAlbertForSequenceClassification
            from tensorflow.keras.layers import Input
            from tensorflow.keras.models import Model
            from tensorflow.keras.utils import to_categorical
            train_seq_feat, train_stat_feat,train_y= data_oversampling(train_seq_feat,train_stat_feat,train_y)
            target_rate = len(target_y[target_y==1])/len(target_y)
            new_data, train_stat_feat = gen_eda(train_seq_feat.tolist(), train_stat_feat, train_y.tolist(), 0.1, 3)
            # new_data, statics_feat = gen_eda(seq_feat.tolist(), statics_feat, y.tolist(), 0.1, i)
            train_seq_feat = new_data["seq"]
            train_y = new_data["bug"]
            maxlen = 512
            del new_data

            tokenizer = AlbertTokenizer.from_pretrained("albert-base-v2")
            train_seq_feat = train_seq_feat.apply(lambda x: tokenizer.encode(x,pad_to_max_length=True))
            target_seq_feat = target_seq_feat.apply(lambda x: tokenizer.encode(x,pad_to_max_length=True))

            train_seq_feat = list(train_seq_feat)
            target_seq_feat = list(target_seq_feat)

            valX = []
            for i in range(len(target_seq_feat)):
                valX.append(target_seq_feat[i][:maxlen])
            target_seq_feat = np.array(valX)
            del valX

            trainX = []
            for i in range(len(train_seq_feat)):
                trainX.append(train_seq_feat[i][:maxlen])
            train_seq_feat = np.array(trainX)
            del trainX
            #input = Input(shape=(maxlen,),batch_size=64, dtype="int32")
            input = Input(shape=(maxlen,),batch_size=64,dtype="int32")
            output = TFAlbertForSequenceClassification.from_pretrained("albert-base-v2")(input)[0]
            print(output)
            model = Model(input,output)
            model.compile(loss="categorical_crossentropy", optimizer="adam",
                          metrics=[km.f1_score(), km.binary_recall(), km.binary_precision()])
            train_y = to_categorical(train_y,2)
            targety = to_categorical(target_y,2)

            modelcheck = ModelCheckpoint("../data/experiment_results/" + project_name + ".h5",save_best_only=True)

            callbacks = [early,modelcheck]
            model.fit(train_seq_feat,train_y,batch_size=64,validation_data=(target_seq_feat,targety),
                      epochs=5,callbacks=callbacks)
            pred = model.predict(target_seq_feat,batch_size=64).argmax(-1)
            f1 = f1_score(target_y,pred)
            print(f1)
            print(output)
        if mode=="WPDP":
            res[pre_project_name] = [f1,precision,recall]
        else:
            res[project_name] = [f1,precision,recall]
    df = pd.DataFrame(res)
    df.to_csv("../data/experiment_results/RQ2/b/"+mode+"_"+model+"_"+".csv",index=False)


# RQ3: Is bert embedding method better than word2vec on both WPDP and CPDP experiment
# embedding method: "albert" and "word2vec"
# models: "textcnn" and "bilstm+attention"
# mode: "WPDP" and "CPDP"

def experiment_RQ3(mode="WPDP",datatype="tokens",embedding="word2vec", model="textcnn"):
    dataset,dataset_list = dataset_generation(mode=mode,datatype=datatype)
    res = {}
    vocab = WordVocab.load_vocab("./pretrained_models/bert/vocab.txt")
    tokenEmb, posEmb = load_bert_weight(max_feature)
    for project_name in dataset_list:
        if mode=="WPDP":
            pre_project_name = dataset[project_name][0]
            train_seq_feat, train_stat_feat, train_y = dataset[project_name][1][0], dataset[project_name][1][1], \
                                                       dataset[project_name][1][2]
            target_seq_feat, target_stat_feat, target_y = dataset[project_name][2][0], dataset[project_name][2][1], \
                                                          dataset[project_name][2][2]
        else:
            train_seq_feat, train_stat_feat, train_y = dataset[project_name][0][0], dataset[project_name][0][1], \
                                                       dataset[project_name][0][2]
            target_seq_feat, target_stat_feat, target_y = dataset[project_name][1][0], dataset[project_name][1][1], \
                                                          dataset[project_name][1][2]
        train_seq_feat, train_stat_feat, train_y = data_oversampling(train_seq_feat, train_stat_feat, train_y)
        train_seq_feat, train_stat_feat, train_y = data_oversampling(train_seq_feat, train_stat_feat, train_y)
        if mode == "WPDP":
            new_data, train_stat_feat = gen_eda(train_seq_feat.tolist(), train_stat_feat, train_y.tolist(), 0.1, 3)
            # new_data, statics_feat = gen_eda(seq_feat.tolist(), statics_feat, y.tolist(), 0.1, i)
            train_seq_feat = new_data["seq"]
            train_y = new_data["bug"]

        if embedding=="word2vec":

            tokenizer = Tokenizer(num_words=max_feature, lower=False)
            tokenizer.fit_on_texts(list(train_seq_feat) + list(target_seq_feat))
            word_index = tokenizer.word_index

            train_seq_feat = tokenizer.texts_to_sequences(list(train_seq_feat))
            train_seq_feat = pad_sequences(train_seq_feat, maxlen=maxlen)

            target_seq_feat = tokenizer.texts_to_sequences(list(target_seq_feat))
            target_seq_feat = pad_sequences(target_seq_feat, maxlen=maxlen)

            with open("./data/embedding_index.pkl", "rb") as f:
                embedding_index = pickle.load(f)
            embedding_matrix = build_matrix(word_index, embedding_index)
            if model == "textcnn":
                # baseline: textCNN for classification
                f1, precision, recall = textcnn(train_x=train_seq_feat, train_y=train_y,
                                                    vocab=tokenizer.index_word, val_x=target_seq_feat,
                                                    val_y=target_y, embedding=embedding_matrix,maxlen=maxlen,mode=embedding,trainable=False)
            else:



                f1, precision, recall = bilstm_att_model(embedding,train_seq_feat, train_y, target_seq_feat, target_y,tokenizer.word_index,
                                                              64, 2,embedding=embedding_matrix,trainable=False)
        else:

            #del new_data
            print("processing begin...")
            train_seq_feat = train_seq_feat.apply(lambda x: tokenize(x, 512, max_feature, vocab))
            target_seq_feat = target_seq_feat.apply(lambda x: tokenize(x, 512, max_feature, vocab))
            print("processing finished")
            train_seq_feat = np.array(list(train_seq_feat))
            target_seq_feat = np.array(list(target_seq_feat))
            train_posEmb = np.expand_dims(posEmb, 0).repeat(train_seq_feat.shape[0], axis=0)
            target_posEmbd = np.expand_dims(posEmb, 0).repeat(target_seq_feat.shape[0], axis=0)
            if model=="textcnn":
                f1, precision, recall = textcnn(train_x=train_seq_feat, train_y=train_y,
                                   vocab=max_feature, val_x=target_seq_feat,
                                   val_y=target_y,embedding=None,maxlen=maxlen,
                                   tokenEmb=tokenEmb,train_posEmb=train_posEmb,
                                   target_posEmb=target_posEmbd,mode="bert",trainable=False)
            else:
                f1, precision, recall = bilstm_att_model("bert",train_seq_feat,train_y,target_seq_feat,target_y,max_feature,64,2,None,tokenEmb,train_posEmb,target_posEmbd,trainable=False)
            print([f1,precision,recall])
        res[project_name] = [round(f1,2),round(precision,2),round(recall,2)]
    df = pd.DataFrame(res)
    df.to_csv("./data/experiment_results/RQ3/"+embedding+"_"+mode+"_"+model+".csv",index=False)
#experiment_RQ3(mode="CPDP",datatype="tokens",embedding="bert")
#experiment_RQ3(mode="CPDP",datatype="tokens",embedding="bert",model="bilstm")
#exit()
# RQ4:investigation about the coverage of sequence length.
#(1000,90%),(550,80%),(350,70%),(250,60%),(180,50%)--->tokens
#(200,90%),(120,80%),(80,70%),(58,60%),(44,50%)---->ast nodes
def experiment_RQ4(mode,datatype="tokens"):
    """
    Strategy: we set the parameter of 'max_len' of embedding layer from 50%~90%,
              since the length of some instances are over 20000,
              we take the length of the coverage from 50%~90%. shown above. we use TextCNN as model.
    Args:
        mode: WPDP or CPDP
        datatype: "tokens" or "ast"

    what function work: training on different coverage of length for 50% to 90% for both ast and tokens type sequences
    """
    dataset, dataset_list = dataset_generation(mode,datatype)
    res = {}
    if datatype=="tokens":
        percent_map = pd.read_csv("./data/experiment_results/RQ1/WPDP_tokens_coverage.csv")
    else:
        percent_map = pd.read_csv("./data/experiment_results/RQ1/WPDP_ast_coverage.csv")
    mapping = {0:0.5,1:0.6,2:0.7,3:0.8,4:0.9}

    for project_name in dataset_list:
        res_in = {}
        count = 0
        for percent in percent_map[project_name]:
            if mode=="WPDP":
                pre_project_name = dataset[project_name][0]
                train_seq_feat, train_stat_feat, train_y = dataset[project_name][1][0], dataset[project_name][1][1], \
                                                           dataset[project_name][1][2]
                target_seq_feat, target_stat_feat, target_y = dataset[project_name][2][0], dataset[project_name][2][1], \
                                                              dataset[project_name][2][2]
            else:
                train_seq_feat, train_stat_feat, train_y = dataset[project_name][0][0], dataset[project_name][0][1], \
                                                           dataset[project_name][0][2]
                target_seq_feat, target_stat_feat, target_y = dataset[project_name][1][0], dataset[project_name][1][1], \
                                                              dataset[project_name][1][2]

            # imbalance
            train_seq_feat, train_stat_feat, train_y = data_oversampling(train_seq_feat,train_stat_feat,train_y)
            new_data, train_stat_feat = gen_eda(train_seq_feat.tolist(), train_stat_feat, train_y.tolist(), 0.1, 3)
            # new_data, statics_feat = gen_eda(seq_feat.tolist(), statics_feat, y.tolist(), 0.1, i)
            train_seq_feat = new_data["seq"]
            train_y = new_data["bug"]
            del new_data
            max_len = percent
            #preprocessing
            tokenizer = Tokenizer(num_words=max_feature, lower=False)
            tokenizer.fit_on_texts(list(train_seq_feat) + list(target_seq_feat))
            word_index = tokenizer.word_index

            train_seq_feat = tokenizer.texts_to_sequences(list(train_seq_feat))
            train_seq_feat = pad_sequences(train_seq_feat, maxlen=max_len)

            target_seq_feat = tokenizer.texts_to_sequences(list(target_seq_feat))
            target_seq_feat = pad_sequences(target_seq_feat, maxlen=max_len)

            embedding_matrix = embedding_matrix_generation(word_index)

            f1, precsion, recall = textcnn(train_x=train_seq_feat, train_y=train_y,
                               vocab=tokenizer.index_word, val_x=target_seq_feat,
                               val_y=target_y, embedding=embedding_matrix,maxlen=max_len,
                                           tokenEmb=None,train_posEmb=None,target_posEmb=None,mode="word2vec")
            #project_name = project_name.split("-")[0]
            project_percent = project_name+str(mapping[count])
            res_in[project_percent] = [f1,precsion,recall]
            print(project_percent)
            count += 1
        res[project_name] = res_in
        print(project_name)
    with open("./data/experiment_results/RQ4/"+mode+"_"+datatype+".pkl","wb") as f:
        pickle.dump(res,f)

# optimal parameter: hidden size, num_layer, NOTE: this RQ question is
# ready for transformers the other data completion. (tentative)
def experiment_RQ5(mode="WPDP",datatype="ast",im=False):
    dataset, dataset_list = dataset_generation(mode=mode,datatype="tokens")
    res = {}
    count = 0
    vocab = WordVocab.load_vocab("../model_files/vocab.txt")
    tokenEmb, posEmb = load_bert_weight(max_feature)
    for project_name in dataset_list:
        res_in = {}
        for num in [8,16,32,48,64,128,256]:
            if mode == "WPDP":
                pre_project_name = dataset[project_name][0]
                train_seq_feat, train_stat_feat, train_y = dataset[project_name][1][0], dataset[project_name][1][1], \
                                                           dataset[project_name][1][2]
                target_seq_feat, target_stat_feat, target_y = dataset[project_name][2][0], dataset[project_name][2][1], \
                                                              dataset[project_name][2][2]
            else:
                train_seq_feat, train_stat_feat, train_y = dataset[project_name][0][0], dataset[project_name][0][1], \
                                                           dataset[project_name][0][2]
                target_seq_feat, target_stat_feat, target_y = dataset[project_name][1][0], dataset[project_name][1][1], \
                                                              dataset[project_name][1][2]

            train_seq_feat, train_stat_feat, train_y = data_oversampling(train_seq_feat, train_stat_feat, train_y)
            max_len = 512
            print("processing begin..")
            train_seq_feat = train_seq_feat.apply(lambda x: tokenize(x, 512, max_feature, vocab))
            target_seq_feat = target_seq_feat.apply(lambda x: tokenize(x, 512, max_feature, vocab))
            print("processing finished")
            train_seq_feat = np.array(list(train_seq_feat))
            target_seq_feat = np.array(list(target_seq_feat))
            train_posEmb = np.expand_dims(posEmb, 0).repeat(train_seq_feat.shape[0], axis=0)
            target_posEmbd = np.expand_dims(posEmb, 0).repeat(target_seq_feat.shape[0], axis=0)
            pred = bert_lstm(train_seq_feat, train_y, target_seq_feat, target_y,
                             tokenEmb, train_posEmb, target_posEmbd, max_feature,hidden_size=num)
            f1 = f1_score(target_y, pred)
            precision = precision_score(target_y, pred)
            recall = recall_score(target_y, pred)
            print([f1,precision,recall],num)
            project_nam = project_name.split("-")[0]
            res_in[project_nam+str(num)] = [f1,precision,recall]
        print(count)
        count += 1
        res[project_name] = res_in
    with open("../data/experiment_results/RQ5/hidden.pkl", "wb") as f:
        pickle.dump(res,f)

if __name__ == "__main__":
    experiment_RQ1(datatype="tokens")


