from torch import nn
from torch.functional import Tensor
import submodels

################################## RNN
class GruEncoder(submodels.Encoder):
    def __init__(self,in_dim,emb_dim,num_hiddens,num_layers,dropout=0,**kwargs):
        super(GruEncoder,self).__init__(**kwargs)
        self.embdding=nn.Embedding(in_dim,emb_dim)
        self.rnn=nn.GRU(emb_dim,num_hiddens,num_layers,dropout=dropout)

    def forward(self,X:Tensor,*args):
        X=self.embdding(X)
        # 更改数据维度为seq_len,batch_size,features
        X=X.permute(1,0,2)
        output,state=self.rnn(X)
        return output,state
##################################