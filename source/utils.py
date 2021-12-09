import time
import os
from torch import nn
from torch import optim
from torch.nn.modules.module import Module
from tqdm.std import tqdm
from settings import *
import json
import pickle as pkl
import re
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader
import torch
from rouge import Rouge
import models

############################### Just run for one time! ###############################
def Preprocess(train_path=DATA_DIR+"train_dataset.csv",test_path=DATA_DIR+"test_dataset.csv"):
    '''
    清理数据、划分验证集后重新保存至新文件
    '''
    
    # 数据清洗
    def _cleanData(data):
        print("数据清洗开始=========================================")
        
        clean_data=[]
        for i,d in tqdm(enumerate(data)):
            res=d
            for pat in PATTERNS_ONCE:
                #################################之后修改
                if("\t" in pat):
                    res=re.sub(pat,"\t",res,1)
                else:
                    res=re.sub(pat,"",res,1)
                ####################################
            for pat in PATTERNS_ANY:
                res=re.sub(pat,"",res)
            
            clean_data.append(res)

        print("数据清洗完毕=========================================")
        return clean_data
    
    # 将处理后的数据保存为json文件
    def _save2Json(data,mode):
        j=0
        
        if mode==2:
            for i in range(len(test_data)): 
                source=test_data[i].split('\t')[1].strip('\n')
                if source!='': 
                    dict_data={"text":source,"summary":'no summary'}#测试集没有参考摘要
                    
                    with open(new_test_path+str(j)+'.json','w+',encoding='utf-8') as f:
                        f.write(json.dumps(dict_data,ensure_ascii=False))
                    j+=1
        
        else:
            for i in range(len(data)):
                if len(data[i].split('\t'))==3:
                    source_seg=data[i].split("\t")[1]
                    traget_seg=data[i].split("\t")[2].strip('\n')
                    
                    if source_seg and traget_seg !='':
                        dict_data={"text":source_seg,"summary":traget_seg}
                        path=new_train_path
                        if mode==1:
                            path= new_val_path  
                        with open(path+str(j)+'.json','w+',encoding='utf-8') as f:
                            f.write(json.dumps(dict_data,ensure_ascii=False)) 
                        j+=1

    
    with open(train_path,'r',encoding='utf-8') as f:
        train_data_all=f.readlines()

    with open(test_path,'r',encoding='utf-8') as f:
        test_data=f.readlines()
    
    # 数据清洗
    train_data_all=_cleanData(train_data_all)
    test_data=_cleanData(test_data)

    # with open("./1.csv",'w',encoding='utf-8') as f:
    #     f.writelines(train_data_all)
    # with open("./2.csv",'w',encoding='utf-8') as f:
    #     f.writelines(test_data)
    # random.shuffle(train_data_all)
    
    # 设置新文件路径
    new_train_path=os.path.join(DATA_DIR,"new_train/")
    new_val_path=os.path.join(DATA_DIR,"new_val/")
    new_test_path=os.path.join(DATA_DIR,"new_test/")

    if not os.path.exists(new_train_path):
        os.makedirs(new_train_path)

    if not os.path.exists(new_val_path):
        os.makedirs(new_val_path)

    if not os.path.exists(new_test_path):
        os.makedirs(new_test_path)

    train_data=train_data_all[:8000] #把训练集重新划分为训练子集和验证子集，保证验证集上loss最小的模型，预测测试集
    val_data=train_data_all[8000:]

    _save2Json(train_data,TRAIN_FALG)
    _save2Json(val_data,VAL_FALG)
    _save2Json(test_data,TEST_FALG)
    

def CountFiles(path):
    '''
    计算目标文件夹json文件数目
    '''
    matcher = re.compile(r'[0-9]+\.json')
    match = lambda name: bool(matcher.match(name))
    names = os.listdir(path)
    n_data = len(list(filter(match, names)))
    return n_data

def BuildVocabCounter(data_dir=DATA_DIR):
    '''
    统计所有词汇，建立词频表
    '''
    from collections import Counter
    
    def GetTokens(path):
        n_data=CountFiles(path)
        summary_words=[]
        source_words=[]
        for i in range(n_data):
            js_data=json.load(open(os.path.join(path,f"{i}.json"),encoding="utf-8"))
            summary=''.join(js_data['summary']).strip()
            summary_words.extend(summary.strip().split(' '))
            
            source=''.join(js_data['text']).strip()
            source_words.extend(source.strip().split(' '))

        return source_words+summary_words

    # print(_count_data(data_dir+"new_train"))
    vocab_counter=Counter()
    vocab_counter.update(t for t in GetTokens(data_dir+"new_train") if t !="")
    vocab_counter.update(t for t in GetTokens(data_dir+"new_val") if t !="")
    vocab_counter.update(t for t in GetTokens(data_dir+"new_test") if t !="")
    # print(vocab_counter.values())

    with open(VOCAB_PATH,"wb") as f:
        pkl.dump(vocab_counter,f)

def MakeVocab(vocab_size=VOCAB_SIZE):
    '''
    建立词典，通过vocab_size设置字典大小，将常用词设置到字典即可，其他生僻词汇用'<unk>'表示
    '''
    with open(VOCAB_PATH,"rb") as f:
        wc=pkl.load(f)
    word2idx, idx2word = {}, {}
    word2idx[PAD_WORD] = 0
    word2idx[UNK_WORD] = 1
    word2idx[BOS_WORD] = 2
    word2idx[EOS_WORD] = 3
    for i, (w, _) in enumerate(wc.most_common(vocab_size), 4):
        word2idx[w] = i
    for w, i in word2idx.items():
        idx2word[i] = w
    
    with open(WORD_IDX_PATH,"wb") as f:
        pkl.dump(word2idx,f)
    with open(IDX_WORD_PATH,"wb") as f:
        pkl.dump(idx2word,f)

def GetNumOfLongestSeq():
    '''
    找到最长的seq长度，用于padding
    '''
    
    def _findInFolders(path,length):
        max_len=0
        for i in range(length):
            js_data=json.load(open(os.path.join(path,f"{i}.json"),encoding="utf-8"))
            l_data=js_data["summary"].split(" ")
            l=len(l_data)
            if(max_len<len(l_data)):
                max_len=l
        return max_len
    
    train_path=os.path.join(DATA_DIR,"new_train/")
    val_path=os.path.join(DATA_DIR,"new_val/")
    test_path=os.path.join(DATA_DIR,"new_test/")

    train_length=CountFiles(train_path)
    val_length=CountFiles(val_path)
    test_length=CountFiles(test_path)
    
    return max(
        _findInFolders(train_path,train_length),
        _findInFolders(val_path,val_length),
        _findInFolders(test_path,test_length))
    


############################### - - ###############################
class TextDataset(Dataset):
    '''生成TensorDataset'''
    def __init__(self,flag,word2id:dict):
        self.word2id=word2id
        self.path=DATA_DIR
        self.flag=flag
        if(flag==TRAIN_FALG):
            self.path+="new_train"
        elif(flag==VAL_FALG):
            self.path+="new_val"
        elif(flag==TEST_FALG):
            self.path+="new_test"
        else:
            raise Exception(f"No this flag:{flag}")
    
    def __len__(self):
        return CountFiles(self.path)

    def __getitem__(self, index):
        source=ReadJson2List(self.path,index)
        summary=ReadJson2List(self.path,index,True)
        # 处理summary中奇怪的问题
        summary=[i for i in summary if (i!='' and i!=' ')]
        # print(summary)
        enc_x=[self.word2id[word] if word in self.word2id.keys() else UNK_NUM for word in source]
        #padding
        enc_x,enc_x_l=PaddingSeq(enc_x,SOURCE_THRESHOLD) 
        
        if(self.flag!=TEST_FALG):
            dec_x=[self.word2id[word] if word in self.word2id.keys() else UNK_NUM for word in summary]
            # decoder输入前面加上BOS、decoder的label最后加上EOS
            y=list(dec_x);y.append(EOS_NUM)
            y,y_l=PaddingSeq(y,SUMMARY_THRESHOLD)

            dec_x.insert(0,BOS_NUM)
            dec_x,dec_x_l=PaddingSeq(dec_x,SUMMARY_THRESHOLD)
        if(self.flag==TEST_FALG):
            return (torch.LongTensor(enc_x),enc_x_l)
        # 返回值依次为：编码器输入，编码器输入有效长度，解码器输入，解码器输入有效长度，标签，标签有效长度
        return (torch.LongTensor(enc_x),enc_x_l),(torch.LongTensor(dec_x),dec_x_l),(torch.LongTensor(y),y_l)

def PaddingSeq(line,threshold):
    """填充文本序列，直接填充转换完的index列表"""
    p_len=len(line)
    if(p_len>threshold):
        if(EOS_NUM in line):
            line[threshold-1]=EOS_NUM
        return line[:threshold],threshold
    return line + [PAD_NUM] * (threshold - len(line)),p_len

def ReadJson2List(dir,i,label=False):
    '''读取单个json文件（一个样本），并按空格分割转换成列表'''
    
    js_data=json.load(open(os.path.join(dir,f"{i}.json"),encoding="utf-8"))
    if label:
        return js_data["summary"].split(" ")
    return js_data["text"].split(" ")


def GetRouge(pred,label):
    '''获取ROUGR-L值'''
    rouge=Rouge()
    rouge_score = rouge.get_scores(pred, label)
    rouge_L_f1 = 0
    rouge_L_p = 0
    rouge_L_r = 0
    for d in rouge_score:
        rouge_L_f1 += d["rouge-l"]["f"]
        rouge_L_p += d["rouge-l"]["p"]
        rouge_L_r += d["rouge-l"]["r"]
    
    return (rouge_L_f1 / len(rouge_score))
    
    print("rouge_f1:%.2f" % (rouge_L_f1 / len(rouge_score)))
    print("rouge_p:%.2f" % (rouge_L_p / len(rouge_score)))
    print("rouge_r:%.2f" % (rouge_L_r / len(rouge_score)))


# 将数据转换为成batch的Tensor，win平台有bug，多进程不能写在函数里
with open(WORD_IDX_PATH,"rb") as f:
        w2i=pkl.load(f)
train_iter=DataLoader(TextDataset(TRAIN_FALG,w2i),shuffle=True,batch_size=BATCH_SZIE,num_workers=8)
val_iter=DataLoader(TextDataset(VAL_FALG,w2i),shuffle=False,batch_size=BATCH_SZIE,num_workers=4)
test_iter=DataLoader(TextDataset(TEST_FALG,w2i),shuffle=False,batch_size=1)

def Train(net:Module,lr=0.01):
    """训练序列到序列模型。"""
    from tqdm import tqdm

    def xavier_init_weights(m):
        if type(m) == nn.Linear:
            nn.init.xavier_uniform_(m.weight)
        if type(m) == nn.GRU:
            for param in m._flat_weights_names:
                if "weight" in param:
                    nn.init.xavier_uniform_(m._parameters[param])

    net.apply(xavier_init_weights)
    net.to(DEVICE)
    optimizer = optim.Adam(net.parameters(), lr=lr)
    loss = models.MaskedSoftmaxCELoss()
    
    # 验证集loss降到10000以下时开始保存每轮更低的参数
    min_loss=10000
    for epoch in range(EPOCHS):
        train_loss=[]
        val_loss=[]

        net.train()
        for batch in tqdm(train_iter):
            (enc_X, enc_x_l), (dec_x, dec_x_l), (y,y_l) = [(x[0].to(DEVICE),x[1].to(DEVICE)) for x in batch]
            
            
            pred, _ = net(enc_X, dec_x, enc_x_l)
            l = loss(pred, y, y_l).sum()
            l.backward()
            
            optimizer.step()       
            optimizer.zero_grad()

            with torch.no_grad():
                train_loss.append(l.item())
            
        # 释放显存
        torch.cuda.empty_cache()

        net.eval()
        with torch.no_grad():
            for batch in tqdm(val_iter):
                (enc_X, enc_x_l), (dec_x, dec_x_l), (y,y_l) = [(x[0].to(DEVICE),x[1].to(DEVICE)) for x in batch]
                pred, _ = net(enc_X, dec_x, enc_x_l)
                l = loss(pred, y, y_l).sum()
                val_loss.append(l.item())

        # 保存模型参数，秒级时间戳保证唯一性
        if(sum(val_loss)<min_loss):
            min_loss=sum(val_loss)
            torch.save(net.state_dict(),PARAM_DIR+str(int(time.time()))+"_GRU.param")
            print(f"saved net with val_loss:{min_loss}")
        print(f"{epoch+1}: train_loss:{sum(train_loss)};val_loss:{sum(val_loss)}")


def TestOneSeq(source:str,net:Module,param_path,max_steps=100,label=None):
    '''测试单个文本，生成摘要'''
    with open(WORD_IDX_PATH,"rb") as f:
        w2i=pkl.load(f)
    with open(IDX_WORD_PATH,"rb") as f:
        i2w=pkl.load(f)
    
    words_list=source.strip().lower().split(" ")
    words_list=[i for i in words_list if (i!='' and i!=' ')]
    # print(words_list)
    id_list=[w2i[word] if word in w2i.keys() else UNK_NUM for word in words_list]
    id_list.append(EOS_NUM)
    print(id_list)
    enc_X,enc_X_l=PaddingSeq(id_list,SOURCE_THRESHOLD)
    enc_X=torch.unsqueeze(torch.tensor(enc_X, dtype=torch.long, device=DEVICE),dim=0)
    enc_X_l=torch.tensor(enc_X_l, dtype=torch.long, device=DEVICE)

    enc_outputs = net.encoder(enc_X, enc_X_l)
    dec_state = net.decoder.init_state(enc_outputs, enc_X_l)
    
    dec_X = torch.unsqueeze(
        torch.tensor([BOS_NUM], dtype=torch.long, device=DEVICE),
        dim=0)

    net.load_state_dict(torch.load(param_path))
    net.eval()
    output_seq, attention_weight_seq = [], []
    for _ in range(max_steps):
        Y, dec_state = net.decoder(dec_X, dec_state)
        dec_X = Y.argmax(dim=2)
        pred = dec_X.squeeze(dim=0).type(torch.int32).item()
        
        if pred == EOS_NUM:
            break
        output_seq.append(pred)
    pred_seq=' '.join([i2w[i] for i in output_seq])
    score=None
    if(label!=None):
        score=GetRouge(pred_seq,label)
    return pred_seq, score

    

def GenSubmisson(net,param_path,max_steps=100):
    '''依据测试集，生成submission文件'''
    import csv
    with open(IDX_WORD_PATH,"rb") as f:
        i2w=pkl.load(f)
    
    net.load_state_dict(torch.load(param_path))
    net.eval()
    res=[]
    count=0
    for enc_X,enc_X_l in tqdm(test_iter):
        enc_X,enc_X_l=enc_X.to(DEVICE),enc_X_l.to(DEVICE)

        dec_X = torch.unsqueeze(
            torch.tensor([BOS_NUM], dtype=torch.long, device=DEVICE),dim=0
            )
        enc_outputs = net.encoder(enc_X, enc_X_l)
        dec_state = net.decoder.init_state(enc_outputs, enc_X_l)
        
        output_seq, attention_weight_seq = [], []
        for _ in range(max_steps):
            Y, dec_state = net.decoder(dec_X, dec_state)
            dec_X = Y.argmax(dim=2)
            pred = dec_X.squeeze(dim=0).type(torch.int32).item()
            if pred == EOS_NUM:
                break
            output_seq.append(pred)
        pred_seq=' '.join([i2w[i] for i in output_seq])
        res.append([str(count),pred_seq])
        count+=1
    
    with open(os.path.join(DATA_DIR, 'submission.csv'),'w+',newline="",encoding='utf-8') as csvfile:
        writer=csv.writer(csvfile,delimiter="\t")   
        writer.writerows(res)


if __name__=='__main__':
    # Preprocess()
    # BuildVocabCounter()
    # MakeVocab(VOCAB_SIZE)
    
    # with open(WORD_IDX_PATH,"rb") as f:
    #     a=pkl.load(f)
    # with open(IDX_WORD_PATH,"rb") as f:
    #     b=pkl.load(f)
    
    # print(a)
    # print(b)
    # print(ReadJson2List(os.path.join(DATA_DIR,"new_test/"),0,True))
    # with open(WORD_IDX_PATH,"rb") as f:
    #     w2i=pkl.load(f)
    # # print(w2i['a'])
    # a=TextDataset(VAL_FALG,w2i)
    # x=a.__getitem__(1)
    
    # print(x)
    # train_iter=DataLoader(TextDataset(VAL_FALG,w2i),shuffle=True,batch_size=128,num_workers=4)
    # Train(models.GetTextSum_GRU(),0.01)
    # GenSubmisson()
    
    # print(
    #     TestOneSeq(
    #     "one-third of phone users would definitely upgrade to a facebook phone - and 73 % think the phone is a ` good idea ' . news of the phone emerged this week , with sources claiming that facebook had hired ex-apple engineers to work on an ` official ' facebook phone . facebook has made several ventures into the mobile market before in partnership with manufacturers such as htc and inq - but a new phone made by ex-apple engineers is rumoured to be in production . the previous ` facebook phone ' - inq 's cloud touch - puts all of your newsfeeds , pictures and other information on a well thought-out homescreen centred around facebook . it 's not the first facebook phone to hit . the market -- the social network giant has previously partnered with inq . and htc to produce facebook-oriented handsets , including phones with a . built-in ` like ' button . details of the proposed phone are scant , but facebook is already making moves into the mobile space with a series of high-profile app acquisitions . after its $ 1 billion purchase of instagram , the social network bought location-based social app glancee and photo-sharing app lightbox . facebook 's smartphone apps have also seen constant and large-scale redesigns , with adverts more prominent with the news feed . the handset is rumoured to be set for a 2013 release . it could be a major hit -- a flash poll of 968 people conducted by myvouchercodes found that 32 % of phone users would upgrade as soon as it became available . the key to its success could be porting apps to mobile -- something facebook is already doing . separate camera and chat apps already separate off some site functions , and third-party apps will shortly be available via a facebook app store . of those polled , 57 % hoped that it would be cheaper than an iphone -- presumably supported by facebook 's advertising . those polled were then asked why they would choose to purchase a facebook phone , if and when one became available , and were asked to select all reasons that applied to them from a list of possible answers . would you ` upgrade ' to a facebook phone ? would you ` upgrade ' to a facebook phone ? now share your opinion . the top five reasons were as follows : . 44 % of people liked the idea of having their mobile phone synced with their facebook account , whilst 41 % said they wanted to be able to use facebook apps on their smartphone . mark pearson , chairman of myvouchercodes.co.uk , said , ` it will be quite exciting to see the first facebook phone when it 's released next year . '",
    #     models.GetTextSum_GRU().to(DEVICE),
    #     os.path.join(PARAM_DIR,"1638704899_GRU.param"),
    #     label=" poll of 968 phone users in uk .   32 % said they would definitely upgrade to a facebook phone .   users hope it might be cheaper than iphone . "
    # )
    # )

    GenSubmisson(
        models.GetTextSum_GRU().to(DEVICE),
        os.path.join(PARAM_DIR,"1638704899_GRU.param")
        )
    