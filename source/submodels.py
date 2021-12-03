
from torch import nn
import torch
from torch.tensor import Tensor
import math

class MaskedSoftmaxCELoss(nn.CrossEntropyLoss):
    """带遮蔽的softmax交叉熵损失函数"""

    def sequence_mask(X, valid_len, value=0):
        """在序列中屏蔽不相关的项。"""
        maxlen = X.size(1)
        mask = torch.arange((maxlen), dtype=torch.float32,
                            device=X.device)[None, :] < valid_len[:, None]
        X[~mask] = value
        return X

    def forward(self, pred, label, valid_len):
        weights = torch.ones_like(label)
        weights = self.sequence_mask(weights, valid_len)
        self.reduction = 'none'
        unweighted_loss = super(MaskedSoftmaxCELoss,
                                self).forward(pred.permute(0, 2, 1), label)
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


    