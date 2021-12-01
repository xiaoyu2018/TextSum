
#------------------ 路径设置 ------------------#
# 数据集目录
DATA_DIR="D:/2021UCAS/高级人工智能/大作业/TextSum/dataset/"
# 模型参数目录
PARAM_DIR="D:/2021UCAS/高级人工智能/大作业/TextSum/params/"
# 词频表地址
VOCAB_PATH="D:/2021UCAS/高级人工智能/大作业/TextSum/dataset/vocab_cnt.pkl"


#------------------ 词典设置 ------------------#
PAD_WORD = '<pad>'
UNK_WORD = '<unk>'
BOS_WORD = '<bos>'
EOS_WORD = '<eos>'

# 数据清理规则
# 顺序莫变！
PATTERNS_ONCE=[
    "by .*? published :.*?\. \| \..*? [0-9]+ \. ",
    "by \. .*? \. ",
    "-lrb- cnn -rrb- -- ",
    "\t(.*?-lrb- .*? -rrb- -- )",
    ]
PATTERNS_ANY=[
    "``|''"
    ]

#------------------ 其他设置 ------------------#
DEVICE="cuda:0"