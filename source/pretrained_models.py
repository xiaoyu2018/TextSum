# 使用预训练模型 t5-small
from torch.nn.modules.module import Module
from transformers import T5Tokenizer, T5ForConditionalGeneration,AdamW
from settings import *
from utils import GetRouge,CountFiles
import os
from torch.utils.data.dataset import TensorDataset
from torch.utils.data.dataloader import DataLoader

def ToTensor(texts,summaries,tokenizer):
    task_prefix="summarize: "
    encoding = tokenizer([task_prefix + sequence for sequence in texts], 
                    padding='longest', 
                    max_length=MAX_SOURCE_LEN, 
                    truncation=True, 
                    return_tensors="pt")
    input_ids, attention_mask = encoding.input_ids, encoding.attention_mask

    target_encoding = tokenizer(summaries, 
                        padding='longest', 
                        max_length=MAX_SUMMARY_LEN, 
                        truncation=True)
    labels = target_encoding.input_ids
    labels = [(i if i != tokenizer.pad_token_id else -100) for i in labels]
    labels = torch.tensor(labels)

    return TensorDataset(input_ids,attention_mask,labels)

def FineTune(net:Module,tokenizer):
    '''微调'''
    
    tset_texts=[]
    tset_summaries=[]
    vset_texts=[]
    vset_summaries=[]
    tset_len=CountFiles(DATA_DIR+"new_train")
    vset_len=CountFiles(DATA_DIR+"new_val")
    for i in range(tset_len):
        text,summary=ReadJson(i,DATA_DIR+"new_train")
        tset_texts.append(text)
        tset_summaries.append(summary)
    for i in range(vset_len):
        text,summary=ReadJson(i,DATA_DIR+"new_val")
        vset_texts.append(text)
        vset_summaries.append(summary)
    print("训练数据已读入内存...")    

    train_iter=DataLoader(
        ToTensor(tset_texts,tset_summaries,tokenizer),
        batch_size=BATCH_SZIE,
        shuffle=True,
        num_workers=4
        )
    val_iter=DataLoader(
        ToTensor(vset_texts,vset_summaries,tokenizer),
        batch_size=BATCH_SZIE,
        shuffle=False,
        num_workers=4
        )

    print("minibatch已生成...") 
       
    print("开始训练模型...")    
    opt=AdamW(net.parameters())
    from tqdm import tqdm
    import time
    min_loss=10
    for epoch in range(EPOCHS):
        train_loss=[]
        val_loss=[]
        net.train()
        for batch in tqdm(train_iter):
            input_ids,attention_mask,labels=[x.to(DEVICE) for x in batch]
            l = net(input_ids=input_ids, attention_mask=attention_mask, labels=labels).loss
            l.backward()
            opt.step()       
            opt.zero_grad()
            with torch.no_grad():
                train_loss.append(l.item())
        
        torch.cuda.empty_cache()
        net.eval()
        with torch.no_grad():
            for batch in tqdm(val_iter):
                input_ids,attention_mask,labels=[x.to(DEVICE) for x in batch]
                l = net(input_ids=input_ids, attention_mask=attention_mask, labels=labels).loss
                val_loss.append(l.item())
        
        if(sum(val_loss)<min_loss):
            min_loss=sum(val_loss)
            torch.save(net.state_dict(),PARAM_DIR+str(int(time.time()))+"_GRU.param")
            print(f"saved net with val_loss:{min_loss}")    
        
        print(f"{epoch+1}: train_loss:{sum(train_loss)};val_loss:{sum(val_loss)}")

def TestOneSeq(net,tokenizer,text, target=None):
    '''生成单个样本的摘要'''
    net.eval()
    
    text = str(text).replace('\n', '')
    input_tokenized = tokenizer.encode(text, return_tensors="pt").to(DEVICE)

    summary_task = torch.tensor([[21603, 10]]).to(DEVICE)
    input_tokenized = torch.cat([summary_task, input_tokenized], dim=-1).to(DEVICE)
    
    summary_ids = net.generate(input_tokenized,
                                    num_beams=NUM_BEAMS,
                                    no_repeat_ngram_size=3,
                                    length_penalty=LEN_PENALTY,
                                    min_length=MIN_LEN,
                                    max_length=MAX_LEN,
                                    early_stopping=True)
    output = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in summary_ids]
    score=-1
    if(target!=None):
        score=GetRouge(output[0],target)
    return output[0],score

# t5
def GetTextSum_T5(name):
    tokenizer=T5Tokenizer.from_pretrained(PARAM_DIR+name)
    net=T5ForConditionalGeneration.from_pretrained(PARAM_DIR+name)
    print(f"{name} 加载完毕")
    return net.to(DEVICE),tokenizer

# # bart
# def GetTextSum_BART():
#     tokenizer=BartTokenizer.from_pretrained(PARAM_DIR+"bart-large-cnn", output_past=True)
#     net=BartForConditionalGeneration.from_pretrained(PARAM_DIR+"bart-large-cnn", output_past=True)
#     print("bart 加载完毕")
#     return (net.to(DEVICE),tokenizer)

def GetPModel(name:str):
    name=name.lower()
    print("正在加载模型")
    if("t5" in name):
        return GetTextSum_T5(name)
    # elif(name=="bart"):
    #     return GetTextSum_BART()
    else:
        raise Exception("该模型未实现！")
    
def ReadJson(i,dir,test=False):
    '''读取单个json文件（一个样本）'''
    import json

    js_data=json.load(open(os.path.join(dir,f"{i}.json"),encoding="utf-8"))
    if test:
        return js_data["text"]
    return js_data["text"],js_data["summary"]

def GenSub(net,tokenizer,param_path=None):
    '''生成submission.csv'''
    import csv
    from tqdm import tqdm
    
    if(param_path!=None):
        net.load_state_dict(torch.load(param_path))
    res=[]
    for i in tqdm(range(1000)):
        text=ReadJson(i,DATA_DIR+"new_test",True)
        summary=TestOneSeq(net,tokenizer,text)[0]
        res.append([str(i),summary])
    with open(os.path.join(DATA_DIR, 'submission.csv'),'w+',newline="",encoding='utf-8') as csvfile:
        writer=csv.writer(csvfile,delimiter="\t")   
        writer.writerows(res)


if __name__=='__main__':
    net,tokenizer=GetPModel("t5-small")
    # res=tokenizer(
    #     ["hello world","hi"], 
    #     return_tensors="pt",
    #     padding='longest', 
    #     max_length=MAX_LEN, 
    #     truncation=True,
    #     )
    # print(res)
    
    # print(TestOneSeq(
    #     net,tokenizer,
    # "one-third of phone users would definitely upgrade to a facebook phone - and 73 % think the phone is a ` good idea ' . news of the phone emerged this week , with sources claiming that facebook had hired ex-apple engineers to work on an ` official ' facebook phone . facebook has made several ventures into the mobile market before in partnership with manufacturers such as htc and inq - but a new phone made by ex-apple engineers is rumoured to be in production . the previous ` facebook phone ' - inq 's cloud touch - puts all of your newsfeeds , pictures and other information on a well thought-out homescreen centred around facebook . it 's not the first facebook phone to hit . the market -- the social network giant has previously partnered with inq . and htc to produce facebook-oriented handsets , including phones with a . built-in ` like ' button . details of the proposed phone are scant , but facebook is already making moves into the mobile space with a series of high-profile app acquisitions . after its $ 1 billion purchase of instagram , the social network bought location-based social app glancee and photo-sharing app lightbox . facebook 's smartphone apps have also seen constant and large-scale redesigns , with adverts more prominent with the news feed . the handset is rumoured to be set for a 2013 release . it could be a major hit -- a flash poll of 968 people conducted by myvouchercodes found that 32 % of phone users would upgrade as soon as it became available . the key to its success could be porting apps to mobile -- something facebook is already doing . separate camera and chat apps already separate off some site functions , and third-party apps will shortly be available via a facebook app store . of those polled , 57 % hoped that it would be cheaper than an iphone -- presumably supported by facebook 's advertising . those polled were then asked why they would choose to purchase a facebook phone , if and when one became available , and were asked to select all reasons that applied to them from a list of possible answers . would you ` upgrade ' to a facebook phone ? would you ` upgrade ' to a facebook phone ? now share your opinion . the top five reasons were as follows : . 44 % of people liked the idea of having their mobile phone synced with their facebook account , whilst 41 % said they wanted to be able to use facebook apps on their smartphone . mark pearson , chairman of myvouchercodes.co.uk , said , ` it will be quite exciting to see the first facebook phone when it 's released next year . '",
    # "poll of 968 phone users in uk .   32 % said they would definitely upgrade to a facebook phone .   users hope it might be cheaper than iphone . "
    # ))
    GenSub(net,tokenizer)

    # opt=AdamW(net.parameters())
    # opt.step()

    # FineTune(net,tokenizer)
