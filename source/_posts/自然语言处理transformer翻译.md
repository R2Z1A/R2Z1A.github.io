---
title: 自然语言处理transformer翻译
date: 2024-06-26 07:52:41
categories: [2024夏]
tags: [人工智能, 自然语言处理]
excerpt: 基于pytorch，实现transformer，线上训练。
thumbnail: /images/blog/nlp_transformer/劝赌.JPG
---

{% notel blue 'fa-solid fa-book' '前言' %}
kaggle代码地址：<https://www.kaggle.com/ayinpasternak/nlp-cj>，如果访问不了可能删了跑路了。
{% endnotel %}

- [概要](#概要)
- [transformer](#transformer)
- [基于Transformer实现机器翻译](#基于transformer实现机器翻译)
  - [导入包](#导入包)
  - [数据处理](#数据处理)
  - [分词](#分词)
  - [数据预处理](#数据预处理)
  - [transformer结构](#transformer结构)
  - [模型实例化、模型训练](#模型实例化模型训练)
  - [模型调用](#模型调用)
  - [保存模型](#保存模型)
- [总结](#总结)

## 概要

本文介绍了transformer模型，并使用transformer模型实现了机器翻译。

## transformer

Transformer 模型是一种深度学习架构，自 2017 年推出以来，彻底改变了自然语言处理 (NLP) 领域。该模型由 Vaswani 等人提出，并已成为 NLP 界最具影响力的模型之一。Transformer 模型通过自注意力机制和并行计算的优势，能够更好地处理长距离依赖关系，提高了模型的训练和推理效率。它在机器翻译、文本摘要、问答系统等多个 NLP 任务中取得了显著的性能提升。

![结构](/images/blog/nlp_transformer/结构.JPG)

transformer由encoder和decoder组成。其中编码器和解码器都堆叠成堆栈。输入经过embedding之后，嵌入位置编码，输入模型。其中多头注意力是上一章节自然语言处理机器翻译中自注意力机制的扩展。

![多头](/images/blog/nlp_transformer/多头.JPG)

多头注意力是一种特殊的自注意力机制，它将查询（Query）、键（Key）和值（Value）投影到多个不同的注意力头（heads）上，每个头都可以独立地计算注意力分数并生成一个输出，最后将这些输出拼接起来形成最终的注意力结果。

多头注意力的工作原理

1. 拆分：将查询、键和值分别通过多个线性变换（通常是通过矩阵乘法实现）映射到多个头上。如果原始维度是d_model，那么每个头的维度将是d_model / h，其中h是头的数量。

2. 计算注意力：在每个头上独立计算注意力分数。这涉及到计算查询和键的点积，然后除以键的维度的平方根（缩放点积注意力），最后应用softmax函数来归一化注意力权重。

3. 加权求和：使用计算出的注意力权重对相应的值进行加权求和，得到每个头的加权值。

4. 拼接：将所有头的加权值拼接起来，并通过另一个线性层进行转换，以产生最终的输出。

## 基于Transformer实现机器翻译

本部分代码notebook kaggle可找。（如果没跑路）

### 导入包

首先确定运行环境，需要注意的是kaggle平台预安装了环境，但是其中的torchtext版本过高，需要降级。

```python
pip install torchtext==0.4.0 
```

```python
import math
import torchtext
import torch
import torch.nn as nn
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from collections import Counter
from torchtext.vocab import Vocab
from torch.nn import TransformerEncoder, TransformerDecoder, TransformerEncoderLayer, TransformerDecoderLayer
import io
import time
import pandas as pd
import numpy as np
import pickle
import tqdm
import sentencepiece as spm
torch.manual_seed(0)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# print(torch.cuda.get_device_name(0)) ## 如果你有GPU，请在你自己的电脑上尝试运行这一套代码

print(device)
```

> cuda

### 数据处理

我们使用JParaCrawl的数据。

```python
df = pd.read_csv('/kaggle/input/nlp-cj-data/zh-ja.bicleaner05.txt', sep='\\t', engine='python', header=None)
trainen = df[2].values.tolist()#[:10000]
trainja = df[3].values.tolist()#[:10000]
# trainen.pop(5972)
# trainja.pop(5972)

# 显示示例
print(trainen[500])
print(trainja[500])
```

> Chinese HS Code Harmonized Code System < HS编码 2905 无环醇及其卤化、磺化、硝化或亚硝化衍生物 HS Code List (Harmonized System Code) for US, UK, EU, China, India, France, Japan, Russia, Germany, Korea, Canada ...
> Japanese HS Code Harmonized Code System < HSコード 2905 非環式アルコール並びにそのハロゲン化誘導体、スルホン化誘導体、ニトロ化誘導体及びニトロソ化誘導体 HS Code List (Harmonized System Code) for US, UK, EU, China, India, France, Japan, Russia, Germany, Korea, Canada ...

### 分词

这里使用预训练模型进行分词。

```python
# 创建SentencePieceProcessor对象
en_tokenizer = spm.SentencePieceProcessor(model_file='/kaggle/input/dataset/spm.en.nopretok.model')
ja_tokenizer = spm.SentencePieceProcessor(model_file='/kaggle/input/dataset/spm.ja.nopretok.model')
```

### 数据预处理

```python
def build_vocab(sentences, tokenizer):
    """
    构建词汇表
    arg:
        sentences：句子列表
        tokenizer：分词器
    """
    counter = Counter() # 创建一个计时器
    for sentence in sentences:
        counter.update(tokenizer.encode(sentence, out_type=str))
    return Vocab(counter, specials=['<unk>', '<pad>', '<bos>', '<eos>'])

# 构造日文词汇表
ja_vocab = build_vocab(trainja, ja_tokenizer)
# 构造英文词汇表
en_vocab = build_vocab(trainen, en_tokenizer)
```

```python
def data_process(ja, en):
    """
    处理训练数据，
    将输入的日语和英语文本转换为张量（tensor）形式
    """
    data = []
    for (raw_ja, raw_en) in zip(ja, en):
        ja_tensor_ = torch.tensor([ja_vocab[token] for token in ja_tokenizer.encode(raw_ja.rstrip("\n"), out_type=str)],
                                  dtype=torch.long)
        en_tensor_ = torch.tensor([en_vocab[token] for token in en_tokenizer.encode(raw_en.rstrip("\n"), out_type=str)],
                                  dtype=torch.long)
    data.append((ja_tensor_, en_tensor_))
    return data
train_data = data_process(trainja, trainen)
```

```python
BATCH_SIZE = 8
PAD_IDX = ja_vocab['<pad>']
BOS_IDX = ja_vocab['<bos>']
EOS_IDX = ja_vocab['<eos>']
def generate_batch(data_batch):
    """
    接受一个数据批次作为输入，
    并对每个数据项进行处理
    """
    ja_batch, en_batch = [], []
    for (ja_item, en_item) in data_batch:
        ja_batch.append(torch.cat([torch.tensor([BOS_IDX]), ja_item, torch.tensor([EOS_IDX])], dim=0))
        en_batch.append(torch.cat([torch.tensor([BOS_IDX]), en_item, torch.tensor([EOS_IDX])], dim=0))
    ja_batch = pad_sequence(ja_batch, padding_value=PAD_IDX)
    en_batch = pad_sequence(en_batch, padding_value=PAD_IDX)
    return ja_batch, en_batch
train_iter = DataLoader(train_data, batch_size=BATCH_SIZE,
                        shuffle=True, collate_fn=generate_batch)
```

### transformer结构

```python
# 导入Transformer模型中的编码器（Encoder）和解码器（Decoder），以及它们的层（Layer）
from torch.nn import (TransformerEncoder, TransformerDecoder,
                      TransformerEncoderLayer, TransformerDecoderLayer)

# TransformerEncoder：用于将输入序列编码为一个连续的向量表示
# TransformerDecoder：用于将编码器的输出向量解码为目标序列
# TransformerEncoderLayer：用于构建编码器的各个层
# TransformerDecoderLayer：用于构建解码器的各个层

class Seq2SeqTransformer(nn.Module):
    def __init__(self, num_encoder_layers: int, num_decoder_layers: int,
                 emb_size: int, src_vocab_size: int, tgt_vocab_size: int,
                 dim_feedforward:int = 512, dropout:float = 0.1):
        super(Seq2SeqTransformer, self).__init__()
        # 创建编码器层，并堆叠成完整的编码器
        encoder_layer = TransformerEncoderLayer(d_model=emb_size, nhead=NHEAD,
                                                dim_feedforward=dim_feedforward)
        self.transformer_encoder = TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        # 创建解码器层，并堆叠成完整的解码器
        decoder_layer = TransformerDecoderLayer(d_model=emb_size, nhead=NHEAD,
                                                dim_feedforward=dim_feedforward)
        self.transformer_decoder = TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)
 
        # 生成器：将编码器的输出映射到目标词汇表的大小
        self.generator = nn.Linear(emb_size, tgt_vocab_size)
        # 词嵌入层：将输入的tokens映射到固定大小的向量空间中
        self.src_tok_emb = TokenEmbedding(src_vocab_size, emb_size)
        self.tgt_tok_emb = TokenEmbedding(tgt_vocab_size, emb_size)
        # 位置编码：为输入的token嵌入添加位置信息，以便模型能够捕捉到序列中的位置关系
        self.positional_encoding = PositionalEncoding(emb_size, dropout=dropout)

    def forward(self, src: Tensor, trg: Tensor, src_mask: Tensor,
                tgt_mask: Tensor, src_padding_mask: Tensor,
                tgt_padding_mask: Tensor, memory_key_padding_mask: Tensor):
        """
        前馈传播
        """
        # 对源序列和目标序列进行词嵌入和位置编码
        src_emb = self.positional_encoding(self.src_tok_emb(src))
        tgt_emb = self.positional_encoding(self.tgt_tok_emb(trg))
        # 使用编码器处理源序列，得到上下文向量
        memory = self.transformer_encoder(src_emb, src_mask, src_padding_mask)
        # 使用解码器将目标序列和上下文向量进行解码，得到输出序列
        outs = self.transformer_decoder(tgt_emb, memory, tgt_mask, None,
                                        tgt_padding_mask, memory_key_padding_mask)
        return self.generator(outs)
 
    def encode(self, src: Tensor, src_mask: Tensor):
        return self.transformer_encoder(self.positional_encoding(
                            self.src_tok_emb(src)), src_mask)

    def decode(self, tgt: Tensor, memory: Tensor, tgt_mask: Tensor):
        return self.transformer_decoder(self.positional_encoding(
                          self.tgt_tok_emb(tgt)), memory,
                          tgt_mask)

# 嵌入添加位置编码（positional encoding）
class PositionalEncoding(nn.Module):
    def __init__(self, emb_size: int, dropout, maxlen: int = 5000):
        super(PositionalEncoding, self).__init__()
        den = torch.exp(- torch.arange(0, emb_size, 2) * math.log(10000) / emb_size)
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)
        pos_embedding = torch.zeros((maxlen, emb_size))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        pos_embedding = pos_embedding.unsqueeze(-2)
 
        self.dropout = nn.Dropout(dropout)
        self.register_buffer('pos_embedding', pos_embedding)
 
    def forward(self, token_embedding: Tensor):
        return self.dropout(token_embedding +
                            self.pos_embedding[:token_embedding.size(0),:])

class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size: int, emb_size):
        super(TokenEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.emb_size = emb_size
    def forward(self, tokens: Tensor):
        return self.embedding(tokens.long()) * math.sqrt(self.emb_size)

def generate_square_subsequent_mask(sz):
    # 创建一个大小为 (sz, sz) 的上三角矩阵，其中对角线及其以上的元素为1，其余为0
    mask = (torch.triu(torch.ones((sz, sz), device=device)) == 1).transpose(0, 1)
    
    # 将 mask 中的 0 替换为负无穷大（表示不可达），1 保持不变（表示可达）
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    
    # 返回处理后的 mask 矩阵
    return mask
 
def create_mask(src, tgt):
    # 获取源序列的长度
    src_seq_len = src.shape[0]
    # 获取目标序列的长度
    tgt_seq_len = tgt.shape[0]
    
    # 生成目标序列的掩码，用于屏蔽后续位置的信息
    tgt_mask = generate_square_subsequent_mask(tgt_seq_len)
    # 创建一个全零矩阵，大小为源序列长度的平方，用于表示源序列的位置信息
    src_mask = torch.zeros((src_seq_len, src_seq_len), device=device).type(torch.bool)
 
    # 创建源序列的填充掩码，将源序列中的填充索引位置标记为True
    src_padding_mask = (src == PAD_IDX).transpose(0, 1)
    # 创建目标序列的填充掩码，将目标序列中的填充索引位置标记为True
    tgt_padding_mask = (tgt == PAD_IDX).transpose(0, 1)
    # 返回源序列掩码、目标序列掩码、源序列填充掩码和目标序列填充掩码
    return src_mask, tgt_mask, src_padding_mask, tgt_padding_mask

```

### 模型实例化、模型训练

```python
SRC_VOCAB_SIZE = len(ja_vocab)
TGT_VOCAB_SIZE = len(en_vocab)
EMB_SIZE = 512
NHEAD = 8
FFN_HID_DIM = 512
BATCH_SIZE = 16
NUM_ENCODER_LAYERS = 3
NUM_DECODER_LAYERS = 3
NUM_EPOCHS = 16
# 创建一个Seq2SeqTransformer对象
transformer = Seq2SeqTransformer(NUM_ENCODER_LAYERS, NUM_DECODER_LAYERS,
                                 EMB_SIZE, SRC_VOCAB_SIZE, TGT_VOCAB_SIZE,
                                 FFN_HID_DIM)
 
for p in transformer.parameters():
    if p.dim() > 1:
        nn.init.xavier_uniform_(p)
 
transformer = transformer.to(device)
 
loss_fn = torch.nn.CrossEntropyLoss(ignore_index=PAD_IDX)
 
optimizer = torch.optim.Adam(
    transformer.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9
)
def train_epoch(model, train_iter, optimizer):
    model.train()
    losses = 0
    for idx, (src, tgt) in  enumerate(train_iter):
        src = src.to(device)
        tgt = tgt.to(device)
 
        tgt_input = tgt[:-1, :]
 
        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt_input)
 
        logits = model(src, tgt_input, src_mask, tgt_mask,
                                src_padding_mask, tgt_padding_mask, src_padding_mask)
 
        optimizer.zero_grad()
 
        tgt_out = tgt[1:,:]
        loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
        loss.backward()
 
        optimizer.step()
        losses += loss.item()
    return losses / len(train_iter)
 
 
def evaluate(model, val_iter):
    model.eval()
    losses = 0
    for idx, (src, tgt) in (enumerate(valid_iter)):
        src = src.to(device)
        tgt = tgt.to(device)
 
        tgt_input = tgt[:-1, :]
 
        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt_input)
 
        logits = model(src, tgt_input, src_mask, tgt_mask,
                       src_padding_mask, tgt_padding_mask, src_padding_mask)
        tgt_out = tgt[1:,:]
        loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
        losses += loss.item()
    return losses / len(val_iter)

for epoch in tqdm.tqdm(range(1, NUM_EPOCHS+1)):
    start_time = time.time()
    train_loss = train_epoch(transformer, train_iter, optimizer)
    end_time = time.time()
    print((f"Epoch: {epoch}, Train loss: {train_loss:.3f}, "
            f"Epoch time = {(end_time - start_time):.3f}s"))
    torch.save({
    'epoch': NUM_EPOCHS,
    'model_state_dict': transformer.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': train_loss,
    }, 'model_checkpoint.tar')
    torch.save(transformer.state_dict(), 'inference_model')

```

### 模型调用

```python
def greedy_decode(model, src, src_mask, max_len, start_symbol):
    # 将输入数据转移到设备上（如GPU）
    src = src.to(device)
    src_mask = src_mask.to(device)
    # 使用模型对源数据进行编码，得到内存表示
    memory = model.encode(src, src_mask)
    # 初始化目标序列，开始符号为start_symbol
    ys = torch.ones(1, 1).fill_(start_symbol).type(torch.long).to(device)
    # 循环生成目标序列，直到达到最大长度或遇到结束符号
    for i in range(max_len-1):
        memory = memory.to(device)
        # 创建目标序列掩码
        memory_mask = torch.zeros(ys.shape[0], memory.shape[0]).to(device).type(torch.bool)
        tgt_mask = (generate_square_subsequent_mask(ys.size(0))
                                    .type(torch.bool)).to(device)
        # 使用模型解码器生成输出
        out = model.decode(ys, memory, tgt_mask)
        out = out.transpose(0, 1)
        # 计算下一个单词的概率分布
        prob = model.generator(out[:, -1])
        # 选择概率最高的单词作为下一个单词
        _, next_word = torch.max(prob, dim = 1)
        next_word = next_word.item()
        # 将下一个单词添加到目标序列中
        ys = torch.cat([ys,
                        torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=0)
        # 如果遇到结束符号，则停止生成
        if next_word == EOS_IDX:
            break
    return ys

def translate(model, src, src_vocab, tgt_vocab, src_tokenizer):
    # 对源文本进行编码，并添加开始和结束符号
    tokens = [BOS_IDX] + [src_vocab.stoi[tok] for tok in src_tokenizer.encode(src, out_type=str)]+ [EOS_IDX]
    num_tokens = len(tokens)
    src = (torch.LongTensor(tokens).reshape(num_tokens, 1) )
    src_mask = (torch.zeros(num_tokens, num_tokens)).type(torch.bool)
    # 使用贪婪解码策略生成目标序列
    tgt_tokens = greedy_decode(model,  src, src_mask, max_len=num_tokens + 5, start_symbol=BOS_IDX).flatten()
    # 将目标序列转换为文本，并移除开始和结束符号
    return " ".join([tgt_vocab.itos[tok] for tok in tgt_tokens]).replace("<bos>", "").replace("<eos>", "")
```

```python
translate(transformer, "HSコード 8515 はんだ付け用、ろう付け用又は溶接用の機器(電気式(電気加熱ガス式を含む。)", ja_vocab, en_vocab, ja_tokenizer)
```

>' ▁H S 代 码 ▁85 15 ▁ 焊 接 焊 接 焊 接 焊 接 焊 接 焊 接 设 备 ( 包 括 电 热 加 热 加 热 加 热 ) 。 '

```python
trainen.pop(5)
```

>'Chinese HS Code Harmonized Code System < HS编码 8515 : 电气(包括电热气体)、激光、其他光、光子束、超声波、电子束、磁脉冲或等离子弧焊接机器及装置,不论是否 HS Code List (Harmonized System Code) for US, UK, EU, China, India, France, Japan, Russia, Germany, Korea, Canada ...'

```python
trainja.pop(5)
```

>'Japanese HS Code Harmonized Code System < HSコード 8515 はんだ付け用、ろう付け用又は溶接用の機器(電気式(電気加熱ガス式を含む。)、レーザーその他の光子ビーム式、超音波式、電子ビーム式、 HS Code List (Harmonized System Code) for US, UK, EU, China, India, France, Japan, Russia, Germany, Korea, Canada ...'

### 保存模型

```python
import pickle
# open a file, where you want to store the data
file = open('en_vocab.pkl', 'wb')
# dump information to that file
pickle.dump(en_vocab, file)
file.close()
file = open('ja_vocab.pkl', 'wb')
pickle.dump(ja_vocab, file)
file.close()

# save model for inference
torch.save(transformer.state_dict(), 'inference_model')

# save model + checkpoint to resume training later
torch.save({
    'epoch': NUM_EPOCHS,
    'model_state_dict': transformer.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': train_loss,
    }, 'model_checkpoint.tar

```

## 总结

Transformer模型主要由编码器（Encoder）和解码器（Decoder）两部分组成。编码器负责将输入序列编码成一个连续的向量表示，解码器则根据这个向量表示生成目标语言的输出序列。

自注意力机制允许模型在计算当前词的表示时，同时考虑到输入序列中的所有词。这使得模型能够更好地捕捉长距离依赖关系，从而提高翻译质量。

多头注意力机制是自注意力机制的扩展，它将自注意力分成多个子空间，每个子空间学习不同的语义信息。这样可以让模型从多个角度理解输入序列，提高翻译的准确性。

位置编码用于解决自注意力机制无法处理序列中词的顺序问题。通过将位置信息编码到输入序列中，模型能够区分不同位置的词，从而保留原始序列的顺序信息。