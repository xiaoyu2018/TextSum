import os
import json
import random
from settings import *


def Preprocess(train_path=DATA_DIR+"train_dataset.csv",test_path=DATA_DIR+"test_dataset.csv"):
    '''
    去掉原始数据中文章描述信息、划分验证集后重新保存至新文件
    '''
    # 将处理后的数据保存为json文件
    def Save2Json(data,mode):
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
    
    random.shuffle(train_data_all)

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

    Save2Json(train_data,1)
    # Save2Json(val_data,2)
    Save2Json(test_data,3)



if __name__=='__main__':
    Preprocess()