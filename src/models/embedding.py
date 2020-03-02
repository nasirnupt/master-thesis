import torch.nn as nn
import torch
import math


class PositionalEmbedding(nn.Module):

    def __init__(self, d_model,max_len=512):
        super().__init__()

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]

class SegmentEmbedding(nn.Embedding):
    def __init__(self, embed_size=512):
        super().__init__(3, embed_size, padding_idx=0)


class TokenEmbedding(nn.Embedding):
    def __init__(self, vocab_size, embed_size=512):
        super().__init__(vocab_size, embed_size, padding_idx=0)


class BERTEmbedding(nn.Module):
    """
    BERT Embedding which is consisted with under features
        1. TokenEmbedding : normal embedding matrix
        2. PositionalEmbedding : adding positional information using sin, cos
        2. SegmentEmbedding : adding sentence segment info, (sent_A:1, sent_B:2)
        sum of all these features are output of BERTEmbedding
    """

    def __init__(self, vocab_size, embed_size, tokenEmb, dropout=0.1):
        """
        :param vocab_size: total vocab size
        :param embed_size: embedding size of token embedding
        :param dropout: dropout rate
        """
        super().__init__()
        self.token = TokenEmbedding(vocab_size=vocab_size, embed_size=embed_size)
        self.token.weight = nn.Parameter(torch.tensor(tokenEmb,dtype=torch.float32),requires_grad=True)
        self.position = PositionalEmbedding(d_model=self.token.embedding_dim)
        #self.segment = SegmentEmbedding(embed_size=self.token.embedding_dim)
        self.embed_size = embed_size
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, sequence):
        x = self.token(sequence) + self.position(sequence)
        return self.dropout(x)

class BERTModel(nn.Module):
    def __init__(self, hidden_size, vocab_size, batch_size,embed_size,tokenEmb):
        super(BERTModel, self).__init__()
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.batch_size = batch_size
        self.embed_size = embed_size
        #self.embedding = BERTEmbedding(self.vocab_size,self.embed_size,tokenEmb)
        self.token = nn.Embedding(vocab_size,self.embed_size)
        self.token.weight = nn.Parameter(torch.tensor(tokenEmb,dtype=torch.float32),requires_grad=True)
        self.posEmb = PositionalEmbedding(embed_size)

        self.lstm = nn.LSTM(self.embed_size,self.hidden_size,1,batch_first=True,bidirectional=True)
        self.fc = nn.Linear(self.hidden_size*2,1)
        self.dropout = nn.Dropout(0.1)
    def forward(self,x):
        print("-------------------")
        token_emb= self.token(x)
        pos_emb = self.posEmb(x)
        embedding = token_emb + pos_emb
        feat, _ = self.lstm(embedding)
        feat = feat[:,-1,:]
        output = self.dropout(feat)
        output = self.fc(output)
        output = torch.softmax(output,dim=-1)
        return output



#####################################################################################
#tensorflow 2.0 version of bert model

