# 使用预训练模型 t5-small
from pickle import FALSE
import re
from transformers import T5Tokenizer, T5ForConditionalGeneration,BartTokenizer, BartForConditionalGeneration
from settings import *
from utils import GetRouge
import os



def FineTune(net,tokenizer):
    pass

def TestOneSeq(net,tokenizer,text, target=None):
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

# t5-small
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
    '''读取单个json文件（一个样本），并按空格分割转换成列表'''
    import json

    js_data=json.load(open(os.path.join(dir,f"{i}.json"),encoding="utf-8"))
    if test:
        return js_data["text"]
    return js_data["text"],js_data["summary"]

def GenSub(net,tokenizer,param_path=None):
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
    net,tokenizer=GetPModel("t5-base")
    
    # print(TestOneSeq(
    #     net,tokenizer,
    # "one-third of phone users would definitely upgrade to a facebook phone - and 73 % think the phone is a ` good idea ' . news of the phone emerged this week , with sources claiming that facebook had hired ex-apple engineers to work on an ` official ' facebook phone . facebook has made several ventures into the mobile market before in partnership with manufacturers such as htc and inq - but a new phone made by ex-apple engineers is rumoured to be in production . the previous ` facebook phone ' - inq 's cloud touch - puts all of your newsfeeds , pictures and other information on a well thought-out homescreen centred around facebook . it 's not the first facebook phone to hit . the market -- the social network giant has previously partnered with inq . and htc to produce facebook-oriented handsets , including phones with a . built-in ` like ' button . details of the proposed phone are scant , but facebook is already making moves into the mobile space with a series of high-profile app acquisitions . after its $ 1 billion purchase of instagram , the social network bought location-based social app glancee and photo-sharing app lightbox . facebook 's smartphone apps have also seen constant and large-scale redesigns , with adverts more prominent with the news feed . the handset is rumoured to be set for a 2013 release . it could be a major hit -- a flash poll of 968 people conducted by myvouchercodes found that 32 % of phone users would upgrade as soon as it became available . the key to its success could be porting apps to mobile -- something facebook is already doing . separate camera and chat apps already separate off some site functions , and third-party apps will shortly be available via a facebook app store . of those polled , 57 % hoped that it would be cheaper than an iphone -- presumably supported by facebook 's advertising . those polled were then asked why they would choose to purchase a facebook phone , if and when one became available , and were asked to select all reasons that applied to them from a list of possible answers . would you ` upgrade ' to a facebook phone ? would you ` upgrade ' to a facebook phone ? now share your opinion . the top five reasons were as follows : . 44 % of people liked the idea of having their mobile phone synced with their facebook account , whilst 41 % said they wanted to be able to use facebook apps on their smartphone . mark pearson , chairman of myvouchercodes.co.uk , said , ` it will be quite exciting to see the first facebook phone when it 's released next year . '",
    # "poll of 968 phone users in uk .   32 % said they would definitely upgrade to a facebook phone .   users hope it might be cheaper than iphone . "
    # ))
    GenSub(net,tokenizer,True)
    
    
    

