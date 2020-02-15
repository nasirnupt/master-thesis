### 0.Abstract (99%)
### 1.Introduction (99%)
+ 1.1 Motivation
### 2.Backgrounds & Related Works(99%)
+ 2.1 Software Defect Prediction
  + 2.1.1 Software Defect Metrics
+ 2.2 Deep learning-based Software Defect Prediction.
+ 2.3 Word Embedding
### 3.Methodology (99%)
+ 3.1 Data Preprocessing
  + 3.1.1 Data Augmentation
+ 3.2 Token Embedding
  + 3.2.1 BERT Pretraining Language Model
  + 3.2.2 Embedding Weight Initialization
+ 3.3 Feature Extraction 
  + 3.3.1 BiLSTM
  + 3.3.2 Attention Mechanism
  + 3.3.3 Global Max Pooling & Global Average Pooling
+ 3.4 Feature Concatenation 
+ 3.5 Training
### 4.Experimental Design (99%)
+ 4.1 Dataset
+ 4.2 Corpus Pretraining
  + 4.2.1 Corpus
  + 4.2.2 Word2vec 
  + 4.2.3 BERT
+ 4.3 Evaluatation Metrics
+ 4.4 Baseline metrics
+ 4.5 Baseline Methods 
+ 4.6 Within-Project Defect Prediction 
+ 4.7 Cross-Project Defect Prediction 
+ 4.8 Parameters Setting 
+ 4.9 Experiment for RQ1  
+ 4.10 Experiment for RQ2
+ 4.11 Experiment for RQ3
+ 4.12 Experiment for RQ4
### 5.Results (99%)
+ 5.1 RQ1:Can Data Augmentation Improve the Performance of Model? 
+ 5.2 RQ2:Can our proposed method better than other models?
+ 5.3 RQ3:Which one are better of our pretrained models?
+ 5.4 RQ4:Length of Coverage comparsion?
### 6.Discussion (99%)
### 7.Conclusion (99%)
### Ackonwledgement (100%)


## Usage
1. Download the whole source code from [google dirver](https://drive.google.com/open?id=17rf5vdkyFoQc9mkaSRTUGRDPK4pL1lgV) or 'pepenero/share/li-jidong'

2. Dependency pakages installation

```
pip install -r requirments.txt
```


### NOTE: Results in this thesis is only for our observation, we do not ensure that you can obtain same results and conclusion as we did.  The reason might be:
1. Limitation number of instance is small. That might be generate usage difference for each turn of training.
2. Our pretrained BERT model. Although it ahieve 6 hihgest F1 scores of our model in RQ2, but it still a little weak since the model are not well-trained. In the future work, we manage to train large corpus to get more robust result.
3. Experiment environment. 

### 1. Corpus pretraining
#### Java source code
Bigcode from the language model. 
#### 1.1 BERT pretraining
Our BERT training use four Nvidia four `RTX 2080 Ti` on Ubuntu system, and it took about 1 day to finish. The pretrained BERT model is availible.  However,our pretrained mini version model are not powerful as we expected. If you would like to train your own BERT model, please refer [Bert-pytorch](https://github.com/Cadene/pretrained-models.pytorch) repository.

#### 1.2 Word2vec pretraining
```
cd pretrained_models
python word2vec_pretraining.py
```

We already have trained Word2vec model in our experiment

### 2. Experiments 

For experiments you can run the following command and generate results, for example run the `RQ1` experiment
```
python main.py RQ1
```
The results will be generated in `data/experiment_results/RQ1/` directory
### 3. Result display
if you want to display the result of `.pkl` dataset, you can run the following command. there are there parameter needed. `Research Question`, `results.pkl`, `metrics`, for example, run the command:
```
python results_display.py RQ1 WPDP.pkl f1
```
to print F1 score of RQ1 of WPDP experiments in markdown.


