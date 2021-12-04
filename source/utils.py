import os
from settings import *
import json
import pickle as pkl
import re
import submodels
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader
import torch

############################### Just run for one time! ###############################
def Preprocess(train_path=DATA_DIR+"train_dataset.csv",test_path=DATA_DIR+"test_dataset.csv"):
    '''
    清理数据、划分验证集后重新保存至新文件
    '''
    import random
    
    # 数据清洗
    def _cleanData(data):
        print("数据清洗开始=========================================")
        
        clean_data=[]
        for i,d in enumerate(data):
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

            if(not (i%300)):
                print(str(i)+"...")
        print("数据清洗完毕=========================================")
        return clean_data
    
    # 将处理后的数据保存为json文件
    def _save2Json(data,mode):
        j=0
        if mode==3:
            for i in range(len(test_data)): 
                source=test_data[i].split('\t')[1].strip('\n')
                
                if source!='': 
                    dict_data={"source":[source],"summary":['no summary']}#测试集没有参考摘要
                    with open(new_test_path+str(j)+'.json','w+',encoding='utf-8') as f:
                        f.write(json.dumps(dict_data,ensure_ascii=False))
                    j+=1
        
        else:
            for i in range(len(data)):
                if len(data[i].split('\t'))==3:
                    source_seg=data[i].split("\t")[1]
                    traget_seg=data[i].split("\t")[2].strip('\n')
                    
                    if source_seg and traget_seg !='':
                        dict_data={"source":[source_seg],"summary":[traget_seg]}
                        path=new_train_path
                        if mode==2:
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
    print(new_train_path)
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
            
            source=''.join(js_data['source']).strip()
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

def MakeVocab(vocab_size):
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
            l_data=js_data["summary"][0].split(" ")
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
        return js_data["summary"][0].split(" ")
    return js_data["source"][0].split(" ")

# 束搜索
def BeamSerch(device,src,model):
    pass


class TextDataset(Dataset):
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


# 将数据转换为成batch的Tensor，win平台有bug，多进程不能写在函数里，凑活用吧
with open(WORD_IDX_PATH,"rb") as f:
        w2i=pkl.load(f)
train_iter=DataLoader(TextDataset(TRAIN_FALG,w2i),shuffle=True,batch_size=128,num_workers=8)
val_iter=DataLoader(TextDataset(VAL_FALG,w2i),shuffle=False,batch_size=64,num_workers=4)
test_iter=DataLoader(TextDataset(TEST_FALG,w2i),shuffle=False,batch_size=64,num_workers=4)


if __name__=='__main__':
    # Preprocess()
    # BuildVocabCounter()
    # MakeVocab(VOCAB_SIZE)
    
    with open(WORD_IDX_PATH,"rb") as f:
        a=pkl.load(f)
    # with open(IDX_WORD_PATH,"rb") as f:
    #     b=pkl.load(f)
    
    print(a)
    # print(b)
    # print(ReadJson2List(os.path.join(DATA_DIR,"new_test/"),0,True))
    # with open(WORD_IDX_PATH,"rb") as f:
    #     w2i=pkl.load(f)
    # # print(w2i['a'])
    # a=TextDataset(VAL_FALG,w2i)
    # x=a.__getitem__(1)
    
    # print(x)
    # train_iter=DataLoader(TextDataset(VAL_FALG,w2i),shuffle=True,batch_size=128,num_workers=4)
    