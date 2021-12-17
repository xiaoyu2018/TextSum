# TextSum
## 0 使用说明
1. 项目相关依赖已写入requirements.txt文件 `pip install -r requirements.txt`
2. 项目使用了`transformers`提供的预训练模型，相关模型、配置文件、词典文件等于[此处](https://huggingface.co/models)下载
3. 运行项目前，于/source/settings.py中修改路径设置为本地实际绝对路径
4. 项目结构：<br/>
TextSum<br/>
--dataset（数据集、词典、词频表）<br/>
--params（预训练模型、模型参数保存文件）<br/>
--source（源代码）<br/>
----go.py（主控函数）<br/>
----pretrained_models.py（预训练模型）<br/>
----models.py（自定义模型）<br/>
----settings.py（项目设置）<br/>
----utils.py（工具函数）<br/>
5. `python go.py` 运行项目，可选命令行参数如下：
    ```
    -h, --help            show this help message and exit
    -p, --preprocess      预处理数据
    -b, --build           建立词频表
    -m, --make            建立词典
    -t 模型名, --train           训练
                            
    -f 模型名, --fine_tune       微调
                            
    -g 模型名 参数路径, --gen             生成submission
                            
    ```
## 1 数据处理
本项目数据处理共分为部分：数据清洗与划分、词典生成、张量转换
+ 数据清洗与划分
  + 使用正则表达式清洗原始数据，去除文本中与任务无关的信息
  + 从原始训练集中划分出验证集
  + 将原始CSV文件转换为逐条文本的JSON文件
+ 词典生成  
统计数据集中出现过的所有单词的词频，取一定数目的高频词生成字典
+ 张量转换  
读取预处理完毕的json文件，进一步处理后将文本数据集转换为成batch的Tensor
## 2 模型结构
本项目使用`pytorch`实现了模型基础结构、自定义损失函数、优化器以及模型训练、验证过程；  
本项目还使用`transformers`提供的预训练模型（bart、t5、pegasus）及函数接口实现了模型的微调与推断  
以下给出部分模型的网络结构
1. GRU编码器-解码器架构网络结构如下：
    ```python
    EncoderDecoder(
    (encoder): GruEncoder(
        (embdding): Embedding(10004, 512)
        (rnn): GRU(512, 256, num_layers=2)
    )
    (decoder): GruDecoder(
        (embdding): Embedding(10004, 512)
        (rnn): GRU(768, 256, num_layers=2)
        (dense): Linear(in_features=256, out_features=10004, bias=True)
    )
    )
    ```
2. t5(small)
    ```python
    T5ForConditionalGeneration(
        (shared): Embedding(32128, 512)
        (encoder): T5Stack(
            (embed_tokens): Embedding(32128, 512)
            (block): ModuleList(
            (0): T5Block(
                (layer): ModuleList(
                (0): T5LayerSelfAttention(
                    (SelfAttention): T5Attention(
                    (q): Linear(in_features=512, out_features=512, bias=False)
                    (k): Linear(in_features=512, out_features=512, bias=False)
                    (v): Linear(in_features=512, out_features=512, bias=False)
                    (o): Linear(in_features=512, out_features=512, bias=False)
                    (relative_attention_bias): Embedding(32, 8)
                    )
                    (layer_norm): T5LayerNorm()
                    (dropout): Dropout(p=0.1, inplace=False)
                )
                (1): T5LayerFF(
                    (DenseReluDense): T5DenseReluDense(
                    (wi): Linear(in_features=512, out_features=2048, bias=False)
                    (wo): Linear(in_features=2048, out_features=512, bias=False)
                    (dropout): Dropout(p=0.1, inplace=False)
                    )
                    (layer_norm): T5LayerNorm()
                    (dropout): Dropout(p=0.1, inplace=False)
                )
                )
            )
            (1): T5Block(
                (layer): ModuleList(
                (0): T5LayerSelfAttention(
                    (SelfAttention): T5Attention(
                    (q): Linear(in_features=512, out_features=512, bias=False)
                    (k): Linear(in_features=512, out_features=512, bias=False)
                    (v): Linear(in_features=512, out_features=512, bias=False)
                    (o): Linear(in_features=512, out_features=512, bias=False)
                    )
                    (layer_norm): T5LayerNorm()
                    (dropout): Dropout(p=0.1, inplace=False)
                )
                (1): T5LayerFF(
                    (DenseReluDense): T5DenseReluDense(
                    (wi): Linear(in_features=512, out_features=2048, bias=False)
                    (wo): Linear(in_features=2048, out_features=512, bias=False)
                    (dropout): Dropout(p=0.1, inplace=False)
                    )
                    (layer_norm): T5LayerNorm()
                    (dropout): Dropout(p=0.1, inplace=False)
                )
                )
            )
            (2): T5Block(
                (layer): ModuleList(
                (0): T5LayerSelfAttention(
                    (SelfAttention): T5Attention(
                    (q): Linear(in_features=512, out_features=512, bias=False)
                    (k): Linear(in_features=512, out_features=512, bias=False)
                    (v): Linear(in_features=512, out_features=512, bias=False)
                    (o): Linear(in_features=512, out_features=512, bias=False)
                    )
                    (layer_norm): T5LayerNorm()
                    (dropout): Dropout(p=0.1, inplace=False)
                )
                (1): T5LayerFF(
                    (DenseReluDense): T5DenseReluDense(
                    (wi): Linear(in_features=512, out_features=2048, bias=False)
                    (wo): Linear(in_features=2048, out_features=512, bias=False)
                    (dropout): Dropout(p=0.1, inplace=False)
                    )
                    (layer_norm): T5LayerNorm()
                    (dropout): Dropout(p=0.1, inplace=False)
                )
                )
            )
            (3): T5Block(
                (layer): ModuleList(
                (0): T5LayerSelfAttention(
                    (SelfAttention): T5Attention(
                    (q): Linear(in_features=512, out_features=512, bias=False)
                    (k): Linear(in_features=512, out_features=512, bias=False)
                    (v): Linear(in_features=512, out_features=512, bias=False)
                    (o): Linear(in_features=512, out_features=512, bias=False)
                    )
                    (layer_norm): T5LayerNorm()
                    (dropout): Dropout(p=0.1, inplace=False)
                )
                (1): T5LayerFF(
                    (DenseReluDense): T5DenseReluDense(
                    (wi): Linear(in_features=512, out_features=2048, bias=False)
                    (wo): Linear(in_features=2048, out_features=512, bias=False)
                    (dropout): Dropout(p=0.1, inplace=False)
                    )
                    (layer_norm): T5LayerNorm()
                    (dropout): Dropout(p=0.1, inplace=False)
                )
                )
            )
            (4): T5Block(
                (layer): ModuleList(
                (0): T5LayerSelfAttention(
                    (SelfAttention): T5Attention(
                    (q): Linear(in_features=512, out_features=512, bias=False)
                    (k): Linear(in_features=512, out_features=512, bias=False)
                    (v): Linear(in_features=512, out_features=512, bias=False)
                    (o): Linear(in_features=512, out_features=512, bias=False)
                    )
                    (layer_norm): T5LayerNorm()
                    (dropout): Dropout(p=0.1, inplace=False)
                )
                (1): T5LayerFF(
                    (DenseReluDense): T5DenseReluDense(
                    (wi): Linear(in_features=512, out_features=2048, bias=False)
                    (wo): Linear(in_features=2048, out_features=512, bias=False)
                    (dropout): Dropout(p=0.1, inplace=False)
                    )
                    (layer_norm): T5LayerNorm()
                    (dropout): Dropout(p=0.1, inplace=False)
                )
                )
            )
            (5): T5Block(
                (layer): ModuleList(
                (0): T5LayerSelfAttention(
                    (SelfAttention): T5Attention(
                    (q): Linear(in_features=512, out_features=512, bias=False)
                    (k): Linear(in_features=512, out_features=512, bias=False)
                    (v): Linear(in_features=512, out_features=512, bias=False)
                    (o): Linear(in_features=512, out_features=512, bias=False)
                    )
                    (layer_norm): T5LayerNorm()
                    (dropout): Dropout(p=0.1, inplace=False)
                )
                (1): T5LayerFF(
                    (DenseReluDense): T5DenseReluDense(
                    (wi): Linear(in_features=512, out_features=2048, bias=False)
                    (wo): Linear(in_features=2048, out_features=512, bias=False)
                    (dropout): Dropout(p=0.1, inplace=False)
                    )
                    (layer_norm): T5LayerNorm()
                    (dropout): Dropout(p=0.1, inplace=False)
                )
                )
            )
            )
            (final_layer_norm): T5LayerNorm()
            (dropout): Dropout(p=0.1, inplace=False)
        )
        (decoder): T5Stack(
            (embed_tokens): Embedding(32128, 512)
            (block): ModuleList(
            (0): T5Block(
                (layer): ModuleList(
                (0): T5LayerSelfAttention(
                    (SelfAttention): T5Attention(
                    (q): Linear(in_features=512, out_features=512, bias=False)
                    (k): Linear(in_features=512, out_features=512, bias=False)
                    (v): Linear(in_features=512, out_features=512, bias=False)
                    (o): Linear(in_features=512, out_features=512, bias=False)
                    (relative_attention_bias): Embedding(32, 8)
                    )
                    (layer_norm): T5LayerNorm()
                    (dropout): Dropout(p=0.1, inplace=False)
                )
                (1): T5LayerCrossAttention(
                    (EncDecAttention): T5Attention(
                    (q): Linear(in_features=512, out_features=512, bias=False)
                    (k): Linear(in_features=512, out_features=512, bias=False)
                    (v): Linear(in_features=512, out_features=512, bias=False)
                    (o): Linear(in_features=512, out_features=512, bias=False)
                    )
                    (layer_norm): T5LayerNorm()
                    (dropout): Dropout(p=0.1, inplace=False)
                )
                (2): T5LayerFF(
                    (DenseReluDense): T5DenseReluDense(
                    (wi): Linear(in_features=512, out_features=2048, bias=False)
                    (wo): Linear(in_features=2048, out_features=512, bias=False)
                    (dropout): Dropout(p=0.1, inplace=False)
                    )
                    (layer_norm): T5LayerNorm()
                    (dropout): Dropout(p=0.1, inplace=False)
                )
                )
            )
            (1): T5Block(
                (layer): ModuleList(
                (0): T5LayerSelfAttention(
                    (SelfAttention): T5Attention(
                    (q): Linear(in_features=512, out_features=512, bias=False)
                    (k): Linear(in_features=512, out_features=512, bias=False)
                    (v): Linear(in_features=512, out_features=512, bias=False)
                    (o): Linear(in_features=512, out_features=512, bias=False)
                    )
                    (layer_norm): T5LayerNorm()
                    (dropout): Dropout(p=0.1, inplace=False)
                )
                (1): T5LayerCrossAttention(
                    (EncDecAttention): T5Attention(
                    (q): Linear(in_features=512, out_features=512, bias=False)
                    (k): Linear(in_features=512, out_features=512, bias=False)
                    (v): Linear(in_features=512, out_features=512, bias=False)
                    (o): Linear(in_features=512, out_features=512, bias=False)
                    )
                    (layer_norm): T5LayerNorm()
                    (dropout): Dropout(p=0.1, inplace=False)
                )
                (2): T5LayerFF(
                    (DenseReluDense): T5DenseReluDense(
                    (wi): Linear(in_features=512, out_features=2048, bias=False)
                    (wo): Linear(in_features=2048, out_features=512, bias=False)
                    (dropout): Dropout(p=0.1, inplace=False)
                    )
                    (layer_norm): T5LayerNorm()
                    (dropout): Dropout(p=0.1, inplace=False)
                )
                )
            )
            (2): T5Block(
                (layer): ModuleList(
                (0): T5LayerSelfAttention(
                    (SelfAttention): T5Attention(
                    (q): Linear(in_features=512, out_features=512, bias=False)
                    (k): Linear(in_features=512, out_features=512, bias=False)
                    (v): Linear(in_features=512, out_features=512, bias=False)
                    (o): Linear(in_features=512, out_features=512, bias=False)
                    )
                    (layer_norm): T5LayerNorm()
                    (dropout): Dropout(p=0.1, inplace=False)
                )
                (1): T5LayerCrossAttention(
                    (EncDecAttention): T5Attention(
                    (q): Linear(in_features=512, out_features=512, bias=False)
                    (k): Linear(in_features=512, out_features=512, bias=False)
                    (v): Linear(in_features=512, out_features=512, bias=False)
                    (o): Linear(in_features=512, out_features=512, bias=False)
                    )
                    (layer_norm): T5LayerNorm()
                    (dropout): Dropout(p=0.1, inplace=False)
                )
                (2): T5LayerFF(
                    (DenseReluDense): T5DenseReluDense(
                    (wi): Linear(in_features=512, out_features=2048, bias=False)
                    (wo): Linear(in_features=2048, out_features=512, bias=False)
                    (dropout): Dropout(p=0.1, inplace=False)
                    )
                    (layer_norm): T5LayerNorm()
                    (dropout): Dropout(p=0.1, inplace=False)
                )
                )
            )
            (3): T5Block(
                (layer): ModuleList(
                (0): T5LayerSelfAttention(
                    (SelfAttention): T5Attention(
                    (q): Linear(in_features=512, out_features=512, bias=False)
                    (k): Linear(in_features=512, out_features=512, bias=False)
                    (v): Linear(in_features=512, out_features=512, bias=False)
                    (o): Linear(in_features=512, out_features=512, bias=False)
                    )
                    (layer_norm): T5LayerNorm()
                    (dropout): Dropout(p=0.1, inplace=False)
                )
                (1): T5LayerCrossAttention(
                    (EncDecAttention): T5Attention(
                    (q): Linear(in_features=512, out_features=512, bias=False)
                    (k): Linear(in_features=512, out_features=512, bias=False)
                    (v): Linear(in_features=512, out_features=512, bias=False)
                    (o): Linear(in_features=512, out_features=512, bias=False)
                    )
                    (layer_norm): T5LayerNorm()
                    (dropout): Dropout(p=0.1, inplace=False)
                )
                (2): T5LayerFF(
                    (DenseReluDense): T5DenseReluDense(
                    (wi): Linear(in_features=512, out_features=2048, bias=False)
                    (wo): Linear(in_features=2048, out_features=512, bias=False)
                    (dropout): Dropout(p=0.1, inplace=False)
                    )
                    (layer_norm): T5LayerNorm()
                    (dropout): Dropout(p=0.1, inplace=False)
                )
                )
            )
            (4): T5Block(
                (layer): ModuleList(
                (0): T5LayerSelfAttention(
                    (SelfAttention): T5Attention(
                    (q): Linear(in_features=512, out_features=512, bias=False)
                    (k): Linear(in_features=512, out_features=512, bias=False)
                    (v): Linear(in_features=512, out_features=512, bias=False)
                    (o): Linear(in_features=512, out_features=512, bias=False)
                    )
                    (layer_norm): T5LayerNorm()
                    (dropout): Dropout(p=0.1, inplace=False)
                )
                (1): T5LayerCrossAttention(
                    (EncDecAttention): T5Attention(
                    (q): Linear(in_features=512, out_features=512, bias=False)
                    (k): Linear(in_features=512, out_features=512, bias=False)
                    (v): Linear(in_features=512, out_features=512, bias=False)
                    (o): Linear(in_features=512, out_features=512, bias=False)
                    )
                    (layer_norm): T5LayerNorm()
                    (dropout): Dropout(p=0.1, inplace=False)
                )
                (2): T5LayerFF(
                    (DenseReluDense): T5DenseReluDense(
                    (wi): Linear(in_features=512, out_features=2048, bias=False)
                    (wo): Linear(in_features=2048, out_features=512, bias=False)
                    (dropout): Dropout(p=0.1, inplace=False)
                    )
                    (layer_norm): T5LayerNorm()
                    (dropout): Dropout(p=0.1, inplace=False)
                )
                )
            )
            (5): T5Block(
                (layer): ModuleList(
                (0): T5LayerSelfAttention(
                    (SelfAttention): T5Attention(
                    (q): Linear(in_features=512, out_features=512, bias=False)
                    (k): Linear(in_features=512, out_features=512, bias=False)
                    (v): Linear(in_features=512, out_features=512, bias=False)
                    (o): Linear(in_features=512, out_features=512, bias=False)
                    )
                    (layer_norm): T5LayerNorm()
                    (dropout): Dropout(p=0.1, inplace=False)
                )
                (1): T5LayerCrossAttention(
                    (EncDecAttention): T5Attention(
                    (q): Linear(in_features=512, out_features=512, bias=False)
                    (k): Linear(in_features=512, out_features=512, bias=False)
                    (v): Linear(in_features=512, out_features=512, bias=False)
                    (o): Linear(in_features=512, out_features=512, bias=False)
                    )
                    (layer_norm): T5LayerNorm()
                    (dropout): Dropout(p=0.1, inplace=False)
                )
                (2): T5LayerFF(
                    (DenseReluDense): T5DenseReluDense(
                    (wi): Linear(in_features=512, out_features=2048, bias=False)
                    (wo): Linear(in_features=2048, out_features=512, bias=False)
                    (dropout): Dropout(p=0.1, inplace=False)
                    )
                    (layer_norm): T5LayerNorm()
                    (dropout): Dropout(p=0.1, inplace=False)
                )
                )
            )
            )
            (final_layer_norm): T5LayerNorm()
            (dropout): Dropout(p=0.1, inplace=False)
        )
        (lm_head): Linear(in_features=512, out_features=32128, bias=False)
        )
    ```

## 3 最终成绩
本项目最终成绩为0.32107609  
    ![](score.png)  
参数设置如下：  
+ 模型：bart-large-cnn
+ 搜索束个数：2
+ 最大序列长度：1024
+ 激活函数：gelu
+ 预测序列最短长度：30
+ 预测序列最长长度：590
+ 是否允许提前停止（预测出`<EOS>`即停止）：是
