import os
from settings import *
import json
import pickle as pkl
import re





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

    _save2Json(train_data,1)
    _save2Json(val_data,2)
    _save2Json(test_data,3)

def BuildVocabCounter(data_dir=DATA_DIR):
    '''
    统计所有词汇，建立词频表
    '''
    from collections import Counter
    

    #计算目标文件夹json文件数目
    def _count_data(path):
        matcher = re.compile(r'[0-9]+\.json')
        match = lambda name: bool(matcher.match(name))
        names = os.listdir(path)
        n_data = len(list(filter(match, names)))
        return n_data
    def GetTokens(path):
        n_data=_count_data(path)
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
    return word2idx, idx2word


if __name__=='__main__':
    # Preprocess()
    # BuildVocabCounter()
    print(MakeVocab(2000))