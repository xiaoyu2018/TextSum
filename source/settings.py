
#------------------ 路径设置 ------------------#
# 数据集目录
DATA_DIR="D:/2021UCAS/高级人工智能/大作业/TextSum/dataset/"
# 模型参数目录
PARAM_DIR="D:/2021UCAS/高级人工智能/大作业/TextSum/params/"
# 词频表地址
VOCAB_PATH="D:/2021UCAS/高级人工智能/大作业/TextSum/dataset/vocab_cnt.pkl"
# 单词->数字
WORD_IDX_PATH="D:/2021UCAS/高级人工智能/大作业/TextSum/dataset/word2idx.pkl"
# 数字->单词
IDX_WORD_PATH="D:/2021UCAS/高级人工智能/大作业/TextSum/dataset/idx2word.pkl"

#------------------ 词典设置 ------------------#
# 特殊符号
PAD_WORD = '<pad>'
UNK_WORD = '<unk>'
BOS_WORD = '<bos>'
EOS_WORD = '<eos>'
PAD_NUM = 0
UNK_NUM = 1
BOS_NUM = 2
EOS_NUM = 3
# 词典大小(拉满就不会出现UNK)
VOCAB_SIZE=100000
# 最长序列长度
MAX_SEQ_LEN=2193
# 读取数据时的标志
TRAIN_FALG=0
VAL_FALG=1
TEST_FALG=2
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