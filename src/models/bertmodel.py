import tensorflow as tf
from transformers import BertTokenizer, TFBertModel
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Input, Flatten, Softmax, concatenate, Dropout, Dense
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.metrics import Recall, Precision
from sklearn.metrics import recall_score, f1_score, precision_score

from utils.experiments import experiment_RQ2_B
experiment_RQ2_B("CPDP",datatype="tokens",model="bilstm")









