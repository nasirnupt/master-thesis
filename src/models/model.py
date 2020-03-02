from tensorflow.keras.layers import Embedding, Dropout, Conv1D, LSTM, MaxPooling1D, concatenate, Flatten, Softmax,Dense, Bidirectional, SpatialDropout1D, GlobalAveragePooling1D, GlobalMaxPooling1D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
import tensorflow.keras.backend as K
from tensorflow.keras.layers import GlobalAveragePooling1D,GlobalMaxPooling1D
import keras
from keras.utils import to_categorical

import keras_metrics as km
from keras.engine.topology import Layer

from keras import initializers, regularizers, constraints, optimizers, layers
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score, recall_score, precision_score
from keras_transformer import get_encoders
from keras_pos_embd.pos_embd import PositionEmbedding
import tensorflow as tf
#from dbn.tensorflow import SupervisedDBNClassification
#from keras_transformer.transformer import EmbeddingRet,EmbeddingSim,TrigPosEmbedding, MultiHeadAttention, FeedForward,LayerNormalization
#from transformers import TFAlbertModel
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler
from transformers.modeling_tf_albert import AlbertConfig, TFAlbertEmbeddings, TFAlbertModel
from tensorflow.keras.layers import concatenate
# all the code implemented keras

# attention layer
class RnnAttentionLayer(layers.Layer):
  def __init__(self, attention_size, drop_rate):
    super().__init__()
    self.attention_size = attention_size
    self.dropout = Dropout(drop_rate, name = "rnn_attention_dropout")
  def build(self, input_shape):
    self.attention_w = self.add_weight(name = "atten_w", shape = (input_shape[-1], self.attention_size), initializer = tf.random_uniform_initializer(), dtype = "float32", trainable = True)
    self.attention_u = self.add_weight(name = "atten_u", shape = (self.attention_size,), initializer = tf.random_uniform_initializer(), dtype = "float32", trainable = True)
    self.attention_b = self.add_weight(name = "atten_b", shape = (self.attention_size,), initializer = tf.constant_initializer(0.1), dtype = "float32", trainable = True)
    super().build(input_shape)
  def call(self, inputs, training):
    x = tf.tanh(tf.add(tf.tensordot(inputs, self.attention_w, axes = 1), self.attention_b))
    x = tf.tensordot(x, self.attention_u, axes = 1)
    x = tf.nn.softmax(x)
    weight_out = tf.multiply(tf.expand_dims(x, -1), inputs)
    final_out = tf.reduce_sum(weight_out, axis = 1)
    drop_out = self.dropout(final_out, training = training)
    return drop_out

from tensorflow.keras import initializers, regularizers, constraints, optimizers, layers

class Attention(tf.keras.layers.Layer):
    def __init__(self, step_dim,
                 W_regularizer=None, b_regularizer=None,
                 W_constraint=None, b_constraint=None,
                 bias=True, **kwargs):
        self.supports_masking = True
        self.init = initializers.get('glorot_uniform')

        self.W_regularizer = regularizers.get(W_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.b_constraint = constraints.get(b_constraint)

        self.bias = bias
        self.step_dim = step_dim
        self.features_dim = 0
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3

        self.W = self.add_weight(shape=(input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        self.features_dim = input_shape[-1]

        if self.bias:
            self.b = self.add_weight(shape=(input_shape[1],),
                                     initializer='zero',
                                     name='{}_b'.format(self.name),
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)
        else:
            self.b = None

        self.built = True

    def compute_mask(self, input, input_mask=None):
        return None

    def call(self, x, mask=None):
        features_dim = self.features_dim
        step_dim = self.step_dim

        eij = K.reshape(K.dot(K.reshape(x, (-1, features_dim)),
                              K.reshape(self.W, (features_dim, 1))), (-1, step_dim))

        if self.bias:
            eij += self.b

        eij = K.tanh(eij)

        a = K.exp(eij)

        if mask is not None:
            a *= K.cast(mask, K.floatx())

        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())
        a = K.expand_dims(a)
        weighted_input = x * a
        return K.sum(weighted_input, axis=1)

    def compute_output_shape(self, input_shape):
        return input_shape[0], self.features_dim


class AttentionLayer(Layer):
    def __init__(self, attention_size=None, **kwargs):
        self.attention_size = attention_size
        super(AttentionLayer, self).__init__(**kwargs)

    def get_config(self):
        config = super().get_config()
        config['attention_size'] = self.attention_size
        return config

    def build(self, input_shape):
        assert len(input_shape) == 3

        self.time_steps = input_shape[1]
        hidden_size = input_shape[2]
        if self.attention_size is None:
            self.attention_size = hidden_size

        self.W = self.add_weight(name='att_weight', shape=(hidden_size, self.attention_size),
                                 initializer='uniform', trainable=True)
        self.b = self.add_weight(name='att_bias', shape=(self.attention_size,),
                                 initializer='uniform', trainable=True)
        self.V = self.add_weight(name='att_var', shape=(self.attention_size,),
                                 initializer='uniform', trainable=True)
        super(AttentionLayer, self).build(input_shape)

    def call(self, inputs):
        self.V = K.reshape(self.V, (-1, 1))
        H = K.tanh(K.dot(inputs, self.W) + self.b)
        score = K.softmax(K.dot(H, self.V), axis=1)
        outputs = K.sum(score * inputs, axis=1)
        return outputs

    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[2]

def scheduler(epoch):
  if epoch < 4:
    return 0.005
  else:
    return 0.005 * tf.math.exp(0.1 * (5 - epoch))
callbacks = [EarlyStopping(), LearningRateScheduler(scheduler)]

def textcnn(mode,train_x,train_y, vocab, val_x, val_y, embedding, maxlen,tokenEmb=None,train_posEmb=None,target_posEmb=None,trainable=True):
    """
    Args:
        train_x: training semantics feature
        train_sx: training statistical feature
        val_x: validation semantics feature
        val_sx: validation statistical feature
        train_y: label of tr#####################################################################################
#tensorflow 2.0 version of bert modelaining set
        val_y: label of validation set
    return:[f1 sorce, precision, recall]
    """

    #embedding method, if albert, choose albert, else word2vec
    if mode == "bert":
        # Embedding layer
        input = Input(shape=(512,), batch_size=64, dtype="int32")
        input2 = Input(shape=(512, 64), batch_size=64, dtype="float32")
        token_embed = Embedding(vocab, 64, weights=[tokenEmb], trainable=trainable)(input)
        # pos_embed = PosEmb(posEmb)(input)
        embeding = tf.add(token_embed, input2)
        embed = Dropout(0.1)(embeding)

    else:
        input = Input(shape=(maxlen,), dtype="float32")
        embed = Embedding(len(vocab) + 1, 300, weights=[embedding], trainable=trainable)(input)

    #textcnn for text classification in other
    cnn1 = Conv1D(32, 2, padding="same", strides=1, activation="relu")(embed)
    cnn1 = MaxPooling1D()(cnn1)

    cnn2 = Conv1D(32, 3, padding="same", strides=1,activation="relu")(embed)
    cnn2 = MaxPooling1D()(cnn2)

    cnn3 = Conv1D(32, 4, padding="same", strides=1,activation="relu")(embed)
    cnn3 = MaxPooling1D()(cnn3)

    features = concatenate([cnn1, cnn2, cnn3], axis=-1)
    

    flat = Flatten()(features)

    # statistitical features
    if mode=="bert":
        drop = Dropout(0.1)(flat)
        main_output = Dense(2, activation="softmax")(drop)
        model = Model(inputs=[input,input2], outputs=main_output)
        model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=[km.f1_score(), km.binary_recall(), km.binary_precision()])
        one_hot_label = keras.utils.to_categorical(train_y,2)
        one_hot_label1 = keras.utils.to_categorical(val_y, 2)
        model.fit([train_x,train_posEmb], one_hot_label, batch_size=64, epochs=5,
                  validation_data=([val_x,target_posEmb],one_hot_label1))
        pred = model.predict([val_x,target_posEmb]).argmax(-1)
    else:

        drop = Dropout(0.1)(flat)
        main_output = Dense(2, activation="softmax")(drop)
        model = Model(inputs=input, outputs=main_output)
        model.compile(loss="categorical_crossentropy", optimizer="adam",
                      metrics=[km.f1_score(), km.binary_recall(), km.binary_precision()])

        one_hot_label = keras.utils.to_categorical(train_y, 2)
        one_hot_label1 = keras.utils.to_categorical(val_y, 2)
        from tensorflow.keras.callbacks import EarlyStopping,ModelCheckpoint

        early = EarlyStopping()
        #check_point = ModelCheckpoint("../data/experiment_results/RQ1/model/")

        model.fit(train_x, one_hot_label, batch_size=64, epochs=5,
                  validation_data=(val_x, one_hot_label1))
        pred = model.predict(val_x,batch_size=64).argmax(-1)
    # metrics calculation
    f1 = f1_score(val_y, pred)
    precision = precision_score(val_y, pred)
    recall = recall_score(val_y, pred)
    return f1, precision, recall

# BiLSTM attention model
def bilstm_att_model(mode,train_x, train_y, val_x, val_y, vocab, hidden_size, class_nums,embedding,tokenEmb=None, train_posEmb=None, target_posEmb=None, trainable=True):
    """
        train_x: training semantics feature
        train_sx: training statistical feature
        val_x: validation semantics feature
        val_sx: validation statistical feature
        train_y: label of training set
        val_y: label of validation set
        hidden_size: complete
    """
    # input layer
    if mode == "bert":
        # Embedding layer
        input = Input(shape=(512,), batch_size=64, dtype="int32")
        input2 = Input(shape=(512, 64), batch_size=64, dtype="float32")
        token_embed = Embedding(vocab, 64, weights=[tokenEmb], trainable=trainable)(input)
        # pos_embed = PosEmb(posEmb)(input)
        embeding = tf.add(token_embed, input2)
        x = Dropout(0.1)(embeding)
    else:
        input = Input(shape=(512,), dtype="int32")
        x = Embedding(len(vocab) + 1, 300, weights=[embedding], trainable=trainable)(input)
    # BiLSTM layer
    x = Bidirectional(LSTM(hidden_size, dropout=0.2, return_sequences=True))(x)
    # Attention layer
    #x = keras.layers.Attention()([x,x])
    #x = Conv1D(16,512,padding="same")(x)
    #x = Flatten()(x)
    x = Attention(512)(x)
    # output layer

    if mode=="bert":
        outputs = Dense(class_nums, activation='softmax')(x)
        # BiLSTM layer
        model = Model(inputs=[input,input2], outputs=outputs)
        one_hot_label = keras.utils.to_categorical(train_y, class_nums)
        one_hot_label1 = keras.utils.to_categorical(val_y, class_nums)
        model.compile(optimizer=Adam(0.005),loss="binary_crossentropy",metrics=[km.binary_precision(), km.binary_recall(), km.f1_score()])
        model.fit([train_x,train_posEmb], one_hot_label,batch_size=64, epochs=5,validation_data=([val_x,target_posEmb],one_hot_label1),callbacks=[EarlyStopping(patience=10)])
        pred = model.predict([val_x,target_posEmb]).argmax(-1)
    else:
        outputs = Dense(class_nums, activation='softmax')(x)
        # BiLSTM layer
        model = Model(inputs=input, outputs=outputs)
        one_hot_label = keras.utils.to_categorical(train_y, class_nums)
        one_hot_label1 = keras.utils.to_categorical(val_y, class_nums)
        model.compile(optimizer=Adam(0.005), loss="binary_crossentropy",
                      metrics=[km.binary_precision(), km.binary_recall(), km.f1_score()])
        model.fit(train_x, one_hot_label, batch_size=64, epochs=5,
                  validation_data=(val_x, one_hot_label1))
        pred = model.predict(val_x).argmax(-1)
    # metrics 计算
    f1 = f1_score(val_y, pred)
    precision = precision_score(val_y, pred)
    recall = recall_score(val_y, pred)
    return f1, precision, recall

# transfomer
def transformer_models(trainX, trainy, valX, valy,embedding,vocab,maxlen,head_num,encoder_num,hidden_dim,project_name):
    #import relevant packages from keras instead of tensorflow.keras to avoid the runtime problem in this model.
    from keras.layers import Input, MaxPooling1D,Flatten,Dense,Embedding,SpatialDropout1D,Dropout,Conv1D
    from keras.models import Model
    from keras.utils import to_categorical
    from keras.callbacks import EarlyStopping,ModelCheckpoint,ReduceLROnPlateau
    import keras
    input = Input(shape=(maxlen,),dtype="int32")

    # embedding
    x = Embedding(len(vocab)+1, 300, weights=[embedding],trainable=False)(input)
    #x = Dropout(0.2)(x)
    print(x.shape)
    #postition embedding
    x = PositionEmbedding(120,300,"add")(x)

    output = get_encoders(encoder_num=encoder_num,input_layer=x,head_num=head_num,hidden_dim=hidden_dim,attention_activation="relu",dropout_rate=0.1)

    # three kind of filters size are 2,3,4
    cnn1 = Conv1D(32, 2, padding="same", strides=1, activation="relu")(x)
    cnn1 = MaxPooling1D()(cnn1)

    cnn2 = Conv1D(32, 3, padding="same", strides=1, activation="relu")(x)
    cnn2 = MaxPooling1D()(cnn2)

    cnn3 = Conv1D(32, 4, padding="same", strides=1, activation="relu")(x)
    cnn3 = MaxPooling1D()(cnn3)

    features = keras.layers.concatenate([cnn1, cnn2, cnn3], axis=-1)

    output = Flatten()(features)
    output = Dropout(0.2)(output)
    output = Dense(2, activation="softmax")(output)
    trainy = to_categorical(trainy,2)
    val_y = to_categorical(valy, 2)
    model = Model(inputs=input, outputs=output)
    path="../data/experiment_results/RQ5/model/"+project_name+str(hidden_dim)+".h5"
    early_stopping = EarlyStopping()
    callbacks = [early_stopping, ModelCheckpoint(path,save_best_only=True)]

    model.compile(optimizer=Adam(0.005),loss="binary_crossentropy",metrics=[km.recall(), km.precision(), km.f1_score()])
    model.fit(trainX, trainy,batch_size=64,epochs=5,callbacks=callbacks,validation_data=(valX,val_y))
    pred = model.predict(valX)
    pred = pred.argmax(-1)

    f1 = f1_score(valy,pred)
    precision = precision_score(valy, pred)
    recall = recall_score(valy, pred)
    return f1, precision, recall

#BiLSTM and TextCNN
def bilstm_textcnn(trainX, trainy, valX, valy, embedding, vocab):
    inputs = Input(shape=(220,), dtype='int32')
    # Embedding
    x = Embedding(len(vocab) + 1, 300, weights=[embedding], trainable=False)(inputs)
    # BiLSTM and attention
    cnn1 = Conv1D(32, 2, padding="same", strides=1, activation="relu")(x)
    cnn1 = MaxPooling1D()(cnn1)

    cnn2 = Conv1D(32, 3, padding="same", strides=1,activation="relu")(x)
    cnn2 = MaxPooling1D()(cnn2)

    cnn3 = Conv1D(32, 4, padding="same", strides=1,activation="relu")(x)
    cnn3 = MaxPooling1D()(cnn3)

    features = concatenate([cnn1, cnn2, cnn3], axis=-1)
    flat = Flatten()(features)

    x = Bidirectional(LSTM(128, dropout=0.2, return_sequences=True))(x)
    # Attention
    x = AttentionLayer(attention_size=128)(x)

    #x = MaxPooling1D(pool_size=4)(x)
    x = concatenate([x,flat], axis=-1)

    x = Dropout(0.2)(x)
    output = Dense(2, activation='softmax')(x)

    #output = Dense(2, activation="softmax")(outputs)
    trainy = to_categorical(trainy,2)
    val_y = to_categorical(valy, 2)
    model = Model(inputs=inputs, outputs=output)
    model.compile(optimizer=Adam(0.005),loss="binary_crossentropy",metrics=[km.recall(), km.precision(), km.f1_score()])
    model.fit(trainX, trainy,batch_size=64,epochs=5)
    pred = model.predict(valX).argmax(-1)
    # metrics
    f1 = f1_score(valy,pred)
    precision = precision_score(valy,pred)
    recall = recall_score(valy,pred)
    return f1, precision, recall


def bert_lstm(train_x,train_y,val_x,val_y, tokenEmb,train_posEmb,target_posEmb,vocab_size=100000,embeding_size=64, hidden_size=64,batch_size=64, dropout=0.1):
    input = Input(shape=(512,),batch_size=batch_size,dtype="int32")
    input2 = Input(shape=(512,64),batch_size=batch_size,dtype="float32")
    token_embed = Embedding(vocab_size,embeding_size,weights=[tokenEmb],trainable=True)(input)
    #pos_embed = PosEmb(posEmb)(input)
    embeding = tf.add(token_embed,input2)
    embeding = Dropout(dropout)(embeding)

    feat = Bidirectional(LSTM(hidden_size,dropout=0.1,return_sequences=True))(embeding)
    att = Attention(512)(feat)
    feat1 = GlobalMaxPooling1D()(feat)
    feat2 = GlobalAveragePooling1D()(feat)
    feat = concatenate([att, feat1,feat2])
    feat = Dropout(0.1)(feat)
    train_y = to_categorical(train_y, 2)
    valy = to_categorical(val_y, 2)
    output = Dense(2,activation="softmax")(feat)
    model = Model([input,input2],output)
    model.compile(optimizer="adam",loss="binary_crossentropy",metrics=[km.f1_score(),km.precision(),km.recall()])
    model.fit([train_x,train_posEmb],train_y,batch_size=batch_size,validation_data=([val_x,target_posEmb],valy),epochs=5,verbose=1,callbacks=[EarlyStopping(patience=5)])
    pred = model.predict([val_x,target_posEmb],batch_size=batch_size).argmax(-1)
    return pred


