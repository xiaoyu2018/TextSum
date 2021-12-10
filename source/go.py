import argparse
import utils
from models import GetModel
import pretrained_models as pm

parser=argparse.ArgumentParser()
parser.add_argument("-p","--preprocess",help="预处理数据",action="store_true")
parser.add_argument("-b","--build",help="建立词频表",action="store_true")
parser.add_argument("-m","--make",help="建立词典",action="store_true")
parser.add_argument("-t","--train",help="训练",type=str)
parser.add_argument("-f","--fine_tune",help="微调",type=str)
parser.add_argument("-g","--gen",help="生成submission",nargs=2,type=str)


args=parser.parse_args()

def main():
    if(args.preprocess):
        print("--------------开始数据预处理--------------")
        try:
            utils.Preprocess()
        except Exception as e:
            print(e)
        print("--------------数据预处理完毕--------------")
        exit(0)
    if(args.build):
        print("--------------开始建立词频表--------------")
        try:
            utils.BuildVocabCounter()
        except Exception as e:
            print(e)
        print("--------------词频表建立完毕--------------")
        exit(0)
    if(args.make):
        print("--------------开始建立字典--------------")
        try:
            utils.MakeVocab()
        except Exception as e:
            print(e)
        print("--------------字典建立完毕--------------")
        exit(0)
    if(args.train):
        
        try:
            net=GetModel(args.train)
            print("--------------开始训练模型--------------")
            utils.Train(net)
            print("--------------模型训练完毕--------------")
        except Exception as e:
            print(e)
        exit(0)
    
    if(args.fine_tune):
        # 最小的模型也train不动。。。
        try:
            net,tkz=pm.GetPModel(args.fine_tune)
            print("--------------开始微调--------------")
            pm.FineTune(net,tkz)
            print("--------------微调完毕--------------")
        except Exception as e:
            print(e)
        exit(0)
    if(args.gen):
        
        net,param_path=args.gen
        
        if(param_path=="x"):
            param_path=None
        try:
            print("--------------开始生成submission--------------")
            if(net=="gru"):
                net=GetModel(net)
                utils.GenSubmisson(net,param_path)
            else:
                net,tkz=pm.GetPModel(net)
                pm.GenSub(net,tkz,param_path)
            
            print("--------------submission生成完毕--------------")
        except Exception as e:
            print(e)
        exit(0)



    print(r"""
___________              __      _________                                  .__                     
\__    ___/___ ___  ____/  |_   /   _____/__ __  _____   _____ _____ _______|__|_______ ___________ 
  |    |_/ __ \\  \/  /\   __\  \_____  \|  |  \/     \ /     \\__  \\_  __ \  \___   // __ \_  __ \
  |    |\  ___/ >    <  |  |    /        \  |  /  Y Y  \  Y Y  \/ __ \|  | \/  |/    /\  ___/|  | \/
  |____| \___  >__/\_ \ |__|   /_______  /____/|__|_|  /__|_|  (____  /__|  |__/_____ \\___  >__|   
             \/      \/                \/            \/      \/     \/               \/    \/       
""")
    print("-h, --help  show help message and exit")

if __name__=='__main__':
    main()