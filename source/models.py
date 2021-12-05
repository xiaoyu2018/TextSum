from torch import nn
import torch
from torch.tensor import Tensor
import math
from settings import *
import utils

class MaskedSoftmaxCELoss(nn.CrossEntropyLoss):
    """带遮蔽的softmax交叉熵损失函数"""

    def _sequence_mask(self, X, valid_len, value=0):
        """ 在序列中屏蔽不相关的项。
            接收valid_len是多个有效长度组成的一维tensor，如[1，2]代表第一个序列有效长度为1，第二个序列有效长度为2
        """
        
        maxlen = X.size(1)
        mask = torch.arange((maxlen), dtype=torch.float32,
                            device=X.device)[None, :] < valid_len[:, None]
        X[~mask] = value
        # 有效长度以外的元素都被置零，不改变原始shape
        return X

    def forward(self, pred, label, valid_len):
        # 不用看标签中的padding的损失
        weights = torch.ones_like(label)
        weights = self._sequence_mask(weights, valid_len)
        self.reduction = 'none'
        unweighted_loss = super().forward(pred.permute(0, 2, 1), label)

        # 把整个序列的loss取平均，最后输出的shape是(batch_size)
        weighted_loss = (unweighted_loss * weights).mean(dim=1)
        return weighted_loss


class Encoder(nn.Module):
    '''编码器接口'''
    def  __init__(self, **kwargs):
        super(Encoder,self).__init__(**kwargs)
    
    def forward(self,X,*args):
        raise NotImplementedError

class Decoder(nn.Module):
    '''编码器接口'''
    def  __init__(self, **kwargs):
        super(Decoder,self).__init__(**kwargs)
    
    # 接收编码器的输出，作为当前步的先验状态
    def init_state(self,enc_outputs,*args):
        raise NotImplementedError
    # state和解码器输入共同作为输入
    # 在一次序列训练中，初始state为编码器输入，之后会不断自我更新
    def forward(self,X,state):
        raise NotImplementedError

class EncoderDecoder(nn.Module):
    '''编码器解码器架构基类'''
    def  __init__(self, encoder:Encoder,decoder:Decoder,**kwargs):
        super(EncoderDecoder,self).__init__(**kwargs)
        self.encoder=encoder
        self.decoder=decoder

    def forward(self,enc_X,dec_X,*args):
        enc_outputs=self.encoder(enc_X,*args)
        dec_state=self.decoder.init_state(enc_outputs)

        return self.decoder(dec_X,dec_state)


################################## RNN
class GruEncoder(Encoder):
    def __init__(self,in_dim,emb_dim,hidden_size,num_layers,dropout=0,**kwargs):
        super(GruEncoder,self).__init__(**kwargs)
        self.embdding=nn.Embedding(in_dim,emb_dim)
        self.rnn=nn.GRU(emb_dim,hidden_size,num_layers,dropout=dropout)

    def forward(self,X:Tensor,*args):
        X=self.embdding(X)
        # 更改数据维度为seq_len,batch_size,features
        X=X.permute(1,0,2)
        output,state=self.rnn(X)
        # shape分别为：
        # (seq_len,batch_size,hidden_size)
        # (num_layers,batch_size,hidden_size)
        return output,state

class GruDecoder(Decoder):
    def __init__(self,in_dim,emb_dim,hidden_size,num_layers,dropout=0,**kwargs):
        super(GruDecoder,self).__init__(**kwargs)
        self.embdding=nn.Embedding(in_dim,emb_dim)
        self.rnn=nn.GRU(emb_dim+hidden_size,hidden_size,num_layers,dropout=dropout)
        self.dense=nn.Linear(hidden_size,VOCAB_SIZE+4)

    def init_state(self, enc_outputs, *args):
        # 取enc的state
        return enc_outputs[1]

    def forward(self,X:Tensor,state:Tensor):
        X=self.embdding(X).permute(1,0,2)
        # 取最后时刻的最后一层
        context=state[-1].repeat(X.shape[0],1,1)
        
        # 虽然state在h0已经传过来了，但是还是把state拼一下,拼到了特征的维度，问题不大
        X_and_context=torch.cat((X,context),2)
        output,state=self.rnn(X_and_context,hx=state)
        output=self.dense(output).permute(1,0,2)
        # shape分别为：
        # (batch_size,seq_len,hidden_size)
        # (num_layers,batch_size,hidden_size)
        return output,state

def GetTextSum_GRU():
    return EncoderDecoder(
        GruEncoder(VOCAB_SIZE+4,512,256,2),
        GruDecoder(VOCAB_SIZE+4,512,256,2)
    )
##################################



if __name__=='__main__':
    encoder=GruEncoder(VOCAB_SIZE+4,512,256,2)
    decoder=GruDecoder(VOCAB_SIZE+4,512,256,2)
    for enc_X,dec_X,y in utils.train_iter:
        print(enc_X[0].shape)
        enc_out=encoder(enc_X[0])
        
        state=decoder.init_state(enc_out)
        output,state=decoder(dec_X[0],state)
        print(output.shape)
        loss_f=MaskedSoftmaxCELoss()
        l=loss_f(output,y[0],y[1])
        print(l)
        
        break
        
