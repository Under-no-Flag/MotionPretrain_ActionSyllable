import numpy as np # 导入 numpy 库
import torch # 导入 torch 库
import torch.nn as nn # 导入 torch.nn 库
d_k = 64 # K(=Q) 维度
d_v = 64 # V 维度
# 定义缩放点积注意力类
class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()
    def forward(self, Q, K, V, attn_mask):
        #------------------------- 维度信息 --------------------------------
        # Q K V [batch_size, n_heads, len_q/k/v, dim_q=k/v] (dim_q=dim_k)
        # attn_mask [batch_size, n_heads, len_q, len_k]
        #----------------------------------------------------------------
        # 计算注意力分数（原始权重）[batch_size，n_heads，len_q，len_k]
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(d_k)
        #------------------------- 维度信息 --------------------------------
        # scores [batch_size, n_heads, len_q, len_k]
        #-----------------------------------------------------------------
        # 使用注意力掩码，将 attn_mask 中值为 1 的位置的权重替换为极小值
        #------------------------- 维度信息 --------------------------------
        # attn_mask [batch_size, n_heads, len_q, len_k], 形状和 scores 相同
        #-----------------------------------------------------------------
        scores.masked_fill_(attn_mask, -1e9)
        # 对注意力分数进行 softmax 归一化
        weights = nn.Softmax(dim=-1)(scores)
        #------------------------- 维度信息 --------------------------------
        # weights [batch_size, n_heads, len_q, len_k], 形状和 scores 相同
        #-----------------------------------------------------------------
        # 计算上下文向量（也就是注意力的输出）, 是上下文信息的紧凑表示
        context = torch.matmul(weights, V)
        #------------------------- 维度信息 --------------------------------
        # context [batch_size, n_heads, len_q, dim_v]
        #-----------------------------------------------------------------
        return context, weights # 返回上下文向量和注意力分数


class MultiHeadAttention(nn.Module):
    def __init__(self,n_heads=8,d_embedding=512):
        super(MultiHeadAttention, self).__init__()

        self.n_heads = n_heads # 多头注意力的头数
        self.W_Q = nn.Linear(d_embedding, d_k * n_heads) # Q的线性变换层
        self.W_K = nn.Linear(d_embedding, d_k * n_heads) # K的线性变换层
        self.W_V = nn.Linear(d_embedding, d_v * n_heads) # V的线性变换层
        self.linear = nn.Linear(n_heads * d_v, d_embedding)
        self.layer_norm = nn.LayerNorm(d_embedding)
    def forward(self, Q, K, V, attn_mask):
        #------------------------- 维度信息 --------------------------------
        # Q K V [batch_size, len_q/k/v, embedding_dim]
        #-----------------------------------------------------------------
        residual, batch_size = Q, Q.size(0) # 保留残差连接
        # 将输入进行线性变换和重塑，以便后续处理
        q_s = self.W_Q(Q).view(batch_size, -1, self.n_heads, d_k).transpose(1,2)
        k_s = self.W_K(K).view(batch_size, -1, self.n_heads, d_k).transpose(1,2)
        v_s = self.W_V(V).view(batch_size, -1, self.n_heads, d_v).transpose(1,2)
        #------------------------- 维度信息 --------------------------------
        # q_s k_s v_s: [batch_size, n_heads, len_q/k/v, d_q=k/v]
        #-----------------------------------------------------------------
        # 将注意力掩码复制到多头 attn_mask: [batch_size, n_heads, len_q, len_k]
        attn_mask = attn_mask.unsqueeze(1).repeat(1, self.n_heads, 1, 1)
        #------------------------- 维度信息 --------------------------------
        # attn_mask [batch_size, n_heads, len_q, len_k]
        #-----------------------------------------------------------------
        # 使用缩放点积注意力计算上下文和注意力权重
        context, weights = ScaledDotProductAttention()(q_s, k_s, v_s, attn_mask)
        #------------------------- 维度信息 --------------------------------
        # context [batch_size, n_heads, len_q, dim_v]
        # weights [batch_size, n_heads, len_q, len_k]
        #-----------------------------------------------------------------
        # 通过调整维度将多个头的上下文向量连接在一起
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.n_heads * d_v)
        #------------------------- 维度信息 --------------------------------
        # context [batch_size, len_q, n_heads * dim_v]
        #-----------------------------------------------------------------
        # 用一个线性层把连接后的多头自注意力结果转换，原始地嵌入维度
        output = self.linear(context)
        #------------------------- 维度信息 --------------------------------
        # output [batch_size, len_q, embedding_dim]
        #-----------------------------------------------------------------
        # 与输入 (Q) 进行残差链接，并进行层归一化后输出
        output = self.layer_norm(output + residual)
        #------------------------- 维度信息 --------------------------------
        # output [batch_size, len_q, embedding_dim]
        #-----------------------------------------------------------------
        return output, weights # 返回层归一化的输出和注意力权重

# 定义逐位置前馈网络类
class PoswiseFeedForwardNet(nn.Module):
    def __init__(self, d_ff=2048,d_embedding=512):
        super(PoswiseFeedForwardNet, self).__init__()
        # 定义一维卷积层 1，用于将输入映射到更高维度
        self.conv1 = nn.Conv1d(in_channels=d_embedding, out_channels=d_ff, kernel_size=1)
        # 定义一维卷积层 2，用于将输入映射回原始维度
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_embedding, kernel_size=1)
        # 定义层归一化
        self.layer_norm = nn.LayerNorm(d_embedding)
    def forward(self, inputs):
        #------------------------- 维度信息 --------------------------------
        # inputs [batch_size, len_q, embedding_dim]
        #----------------------------------------------------------------
        residual = inputs  # 保留残差连接
        # 在卷积层 1 后使用 ReLU 激活函数
        output = nn.ReLU()(self.conv1(inputs.transpose(1, 2)))
        #------------------------- 维度信息 --------------------------------
        # output [batch_size, d_ff, len_q]
        #----------------------------------------------------------------
        # 使用卷积层 2 进行降维
        output = self.conv2(output).transpose(1, 2)
        #------------------------- 维度信息 --------------------------------
        # output [batch_size, len_q, embedding_dim]
        #----------------------------------------------------------------
        # 与输入进行残差链接，并进行层归一化
        output = self.layer_norm(output + residual)
        #------------------------- 维度信息 --------------------------------
        # output [batch_size, len_q, embedding_dim]
        #----------------------------------------------------------------
        return output # 返回加入残差连接后层归一化的结果

# 生成正弦位置编码表的函数，用于在 Transformer 中引入位置信息
def get_sin_enc_table(n_position, embedding_dim):
    #------------------------- 维度信息 --------------------------------
    # n_position: 输入序列的最大长度
    # embedding_dim: 词嵌入向量的维度
    #-----------------------------------------------------------------
    # 根据位置和维度信息，初始化正弦位置编码表
    sinusoid_table = np.zeros((n_position, embedding_dim))
    # 遍历所有位置和维度，计算角度值
    for pos_i in range(n_position):
        for hid_j in range(embedding_dim):
            angle = pos_i / np.power(10000, 2 * (hid_j // 2) / embedding_dim)
            sinusoid_table[pos_i, hid_j] = angle
    # 计算正弦和余弦值
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i 偶数维
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1 奇数维
    #------------------------- 维度信息 --------------------------------
    # sinusoid_table 的维度是 [n_position, embedding_dim]
    #----------------------------------------------------------------
    return torch.FloatTensor(sinusoid_table)  # 返回正弦位置编码表

# 定义填充注意力掩码函数
def get_attn_pad_mask(seq_q, seq_k):
    #------------------------- 维度信息 --------------------------------
    # seq_q 的维度是 [batch_size, len_q]
    # seq_k 的维度是 [batch_size, len_k]
    #-----------------------------------------------------------------
    batch_size, len_q = seq_q.size()
    batch_size, len_k = seq_k.size()
    # 生成布尔类型张量
    pad_attn_mask = seq_k.data.eq(0).unsqueeze(1)  # <PAD>token 的编码值为 0
    #------------------------- 维度信息 --------------------------------
    # pad_attn_mask 的维度是 [batch_size，1，len_k]
    #-----------------------------------------------------------------
    # 变形为与注意力分数相同形状的张量
    pad_attn_mask = pad_attn_mask.expand(batch_size, len_q, len_k)
    #------------------------- 维度信息 --------------------------------
    # pad_attn_mask 的维度是 [batch_size，len_q，len_k]
    #-----------------------------------------------------------------
    return pad_attn_mask

# 定义编码器层类
class EncoderLayer(nn.Module):
    def __init__(self):
        super(EncoderLayer, self).__init__()
        self.enc_self_attn = MultiHeadAttention() # 多头自注意力层
        self.pos_ffn = PoswiseFeedForwardNet() # 位置前馈神经网络层
    def forward(self, enc_inputs, enc_self_attn_mask):
        #------------------------- 维度信息 --------------------------------
        # enc_inputs 的维度是 [batch_size, seq_len, embedding_dim]
        # enc_self_attn_mask 的维度是 [batch_size, seq_len, seq_len]
        #-----------------------------------------------------------------
        # 将相同的 Q，K，V 输入多头自注意力层 , 返回的 attn_weights 增加了头数
        enc_outputs, attn_weights = self.enc_self_attn(enc_inputs, enc_inputs,
                                               enc_inputs, enc_self_attn_mask)
        #------------------------- 维度信息 --------------------------------
        # enc_outputs 的维度是 [batch_size, seq_len, embedding_dim]
        # attn_weights 的维度是 [batch_size, n_heads, seq_len, seq_len]
        # 将多头自注意力 outputs 输入位置前馈神经网络层
        enc_outputs = self.pos_ffn(enc_outputs) # 维度与 enc_inputs 相同
        #------------------------- 维度信息 --------------------------------
        # enc_outputs 的维度是 [batch_size, seq_len, embedding_dim]
        #-----------------------------------------------------------------
        return enc_outputs, attn_weights # 返回编码器输出和每层编码器注意力权重


# 定义编码器类
# n_layers = 6  # 设置 Encoder 的层数
class Encoder(nn.Module):
    def __init__(self, corpus,d_embedding=512, n_layers=6):
        super(Encoder, self).__init__()
        self.src_emb = nn.Embedding(len(corpus.src_vocab), d_embedding) # 词嵌入层
        self.pos_emb = nn.Embedding.from_pretrained( \
          get_sin_enc_table(corpus.src_len+1, d_embedding), freeze=True) # 位置嵌入层
        self.layers = nn.ModuleList(EncoderLayer() for _ in range(n_layers))# 编码器层数
    def forward(self, enc_inputs):
        #------------------------- 维度信息 --------------------------------
        # enc_inputs 的维度是 [batch_size, source_len]
        #-----------------------------------------------------------------
        # 创建一个从 1 到 source_len 的位置索引序列
        pos_indices = torch.arange(1, enc_inputs.size(1) + 1).unsqueeze(0).to(enc_inputs)
        #------------------------- 维度信息 --------------------------------
        # pos_indices 的维度是 [1, source_len]
        #-----------------------------------------------------------------
        # 对输入进行词嵌入和位置嵌入相加 [batch_size, source_len，embedding_dim]
        enc_outputs = self.src_emb(enc_inputs) + self.pos_emb(pos_indices)
        #------------------------- 维度信息 --------------------------------
        # enc_outputs 的维度是 [batch_size, seq_len, embedding_dim]
        #-----------------------------------------------------------------
        # 生成自注意力掩码
        enc_self_attn_mask = get_attn_pad_mask(enc_inputs, enc_inputs)
        #------------------------- 维度信息 --------------------------------
        # enc_self_attn_mask 的维度是 [batch_size, len_q, len_k]
        #-----------------------------------------------------------------
        enc_self_attn_weights = [] # 初始化 enc_self_attn_weights
        # 通过编码器层 [batch_size, seq_len, embedding_dim]
        for layer in self.layers:
            enc_outputs, enc_self_attn_weight = layer(enc_outputs, enc_self_attn_mask)
            enc_self_attn_weights.append(enc_self_attn_weight)
        #------------------------- 维度信息 --------------------------------
        # enc_outputs 的维度是 [batch_size, seq_len, embedding_dim] 维度与 enc_inputs 相同
        # enc_self_attn_weights 是一个列表，每个元素的维度是 [batch_size, n_heads, seq_len, seq_len]
        #-----------------------------------------------------------------
        return enc_outputs, enc_self_attn_weights # 返回编码器输出和编码器注意力权重


# 生成后续注意力掩码的函数，用于在多头自注意力计算中忽略未来信息
def get_attn_subsequent_mask(seq):
    #------------------------- 维度信息 --------------------------------
    # seq 的维度是 [batch_size, seq_len(Q)=seq_len(K)]
    #-----------------------------------------------------------------
    # 获取输入序列的形状
    attn_shape = [seq.size(0), seq.size(1), seq.size(1)]
    #------------------------- 维度信息 --------------------------------
    # attn_shape 是一个一维张量 [batch_size, seq_len(Q), seq_len(K)]
    #-----------------------------------------------------------------
    # 使用 numpy 创建一个上三角矩阵（triu = triangle upper）
    subsequent_mask = np.triu(np.ones(attn_shape), k=1)
    #------------------------- 维度信息 --------------------------------
    # subsequent_mask 的维度是 [batch_size, seq_len(Q), seq_len(K)]
    #-----------------------------------------------------------------
    # 将 numpy 数组转换为 PyTorch 张量，并将数据类型设置为 byte（布尔值）
    subsequent_mask = torch.from_numpy(subsequent_mask).byte()
    #------------------------- 维度信息 --------------------------------
    # 返回的 subsequent_mask 的维度是 [batch_size, seq_len(Q), seq_len(K)]
    #-----------------------------------------------------------------
    return subsequent_mask # 返回后续位置的注意力掩码

# 定义解码器层类
class DecoderLayer(nn.Module):
    def __init__(self):
        super(DecoderLayer, self).__init__()
        self.dec_self_attn = MultiHeadAttention() # 多头自注意力层
        self.dec_enc_attn = MultiHeadAttention()  # 多头自注意力层，连接编码器和解码器
        self.pos_ffn = PoswiseFeedForwardNet() # 位置前馈神经网络层
    def forward(self, dec_inputs, enc_outputs, dec_self_attn_mask, dec_enc_attn_mask):
        #------------------------- 维度信息 --------------------------------
        # dec_inputs 的维度是 [batch_size, target_len, embedding_dim]
        # enc_outputs 的维度是 [batch_size, source_len, embedding_dim]
        # dec_self_attn_mask 的维度是 [batch_size, target_len, target_len]
        # dec_enc_attn_mask 的维度是 [batch_size, target_len, source_len]
        #-----------------------------------------------------------------
        # 将相同的 Q，K，V 输入多头自注意力层
        dec_outputs, dec_self_attn = self.dec_self_attn(dec_inputs, dec_inputs,
                                                        dec_inputs, dec_self_attn_mask)
        #------------------------- 维度信息 --------------------------------
        # dec_outputs 的维度是 [batch_size, target_len, embedding_dim]
        # dec_self_attn 的维度是 [batch_size, n_heads, target_len, target_len]
        #-----------------------------------------------------------------
        # 将解码器输出和编码器输出输入多头自注意力层
        dec_outputs, dec_enc_attn = self.dec_enc_attn(dec_outputs, enc_outputs,
                                                      enc_outputs, dec_enc_attn_mask)
        #------------------------- 维度信息 --------------------------------
        # dec_outputs 的维度是 [batch_size, target_len, embedding_dim]
        # dec_enc_attn 的维度是 [batch_size, n_heads, target_len, source_len]
        #-----------------------------------------------------------------
        # 输入位置前馈神经网络层
        dec_outputs = self.pos_ffn(dec_outputs)
        #------------------------- 维度信息 --------------------------------
        # dec_outputs 的维度是 [batch_size, target_len, embedding_dim]
        # dec_self_attn 的维度是 [batch_size, n_heads, target_len, target_len]
        # dec_enc_attn 的维度是 [batch_size, n_heads, target_len, source_len]
        #-----------------------------------------------------------------
        # 返回解码器层输出，每层的自注意力和解 - 编码器注意力权重
        return dec_outputs, dec_self_attn, dec_enc_attn

#  定义解码器类
# n_layers = 6  # 设置 Decoder 的层数
class Decoder(nn.Module):
    def __init__(self, corpus, d_embedding=512, n_layers=6):
        super(Decoder, self).__init__()
        self.tgt_emb = nn.Embedding(len(corpus.tgt_vocab), d_embedding) # 词嵌入层
        self.pos_emb = nn.Embedding.from_pretrained( \
           get_sin_enc_table(corpus.tgt_len+1, d_embedding), freeze=True) # 位置嵌入层
        self.layers = nn.ModuleList([DecoderLayer() for _ in range(n_layers)]) # 叠加多层



    def forward(self, dec_inputs, enc_inputs, enc_outputs):
        #------------------------- 维度信息 --------------------------------
        # dec_inputs 的维度是 [batch_size, target_len]
        # enc_inputs 的维度是 [batch_size, source_len]
        # enc_outputs 的维度是 [batch_size, source_len, embedding_dim]
        #-----------------------------------------------------------------
        # 创建一个从 1 到 source_len 的位置索引序列
        pos_indices = torch.arange(1, dec_inputs.size(1) + 1).unsqueeze(0).to(dec_inputs)
        #------------------------- 维度信息 --------------------------------
        # pos_indices 的维度是 [1, target_len]
        #-----------------------------------------------------------------
        # 对输入进行词嵌入和位置嵌入相加
        dec_outputs = self.tgt_emb(dec_inputs) + self.pos_emb(pos_indices)
        #------------------------- 维度信息 --------------------------------
        # dec_outputs 的维度是 [batch_size, target_len, embedding_dim]
         #-----------------------------------------------------------------
        # 生成解码器自注意力掩码和解码器 - 编码器注意力掩码
        dec_self_attn_pad_mask = get_attn_pad_mask(dec_inputs, dec_inputs) # 填充位掩码
        dec_self_attn_subsequent_mask = get_attn_subsequent_mask(dec_inputs) # 后续位掩码
        dec_self_attn_mask = torch.gt((dec_self_attn_pad_mask \
                                       + dec_self_attn_subsequent_mask), 0)
        dec_enc_attn_mask = get_attn_pad_mask(dec_inputs, enc_inputs) # 解码器 - 编码器掩码
        #------------------------- 维度信息 --------------------------------
        # dec_self_attn_pad_mask 的维度是 [batch_size, target_len, target_len]
        # dec_self_attn_subsequent_mask 的维度是 [batch_size, target_len, target_len]
        # dec_self_attn_mask 的维度是 [batch_size, target_len, target_len]
        # dec_enc_attn_mask 的维度是 [batch_size, target_len, source_len]
         #-----------------------------------------------------------------
        dec_self_attns, dec_enc_attns = [], [] # 初始化 dec_self_attns, dec_enc_attns
        # 通过解码器层 [batch_size, seq_len, embedding_dim]
        for layer in self.layers:
            dec_outputs, dec_self_attn, dec_enc_attn = layer(dec_outputs, enc_outputs,
                                               dec_self_attn_mask, dec_enc_attn_mask)
            dec_self_attns.append(dec_self_attn)
            dec_enc_attns.append(dec_enc_attn)
        #------------------------- 维度信息 --------------------------------
        # dec_outputs 的维度是 [batch_size, target_len, embedding_dim]
        # dec_self_attns 是一个列表，每个元素的维度是 [batch_size, n_heads, target_len, target_len]
        # dec_enc_attns 是一个列表，每个元素的维度是 [batch_size, n_heads, target_len, source_len]
        #-----------------------------------------------------------------
        # 返回解码器输出，解码器自注意力和解码器 - 编码器注意力权重
        return dec_outputs, dec_self_attns, dec_enc_attns

# 定义 Transformer 模型
class Transformer(nn.Module):
    def __init__(self, corpus,e_embedding=512, d_embedding=512, n_layers=6):
        super(Transformer, self).__init__()
        self.encoder = Encoder(corpus) # 初始化编码器实例
        self.decoder = Decoder(corpus) # 初始化解码器实例
        # 定义线性投影层，将解码器输出转换为目标词汇表大小的概率分布
        self.projection = nn.Linear(d_embedding, len(corpus.tgt_vocab), bias=False)


    def forward(self, enc_inputs, dec_inputs):
        #------------------------- 维度信息 --------------------------------
        # enc_inputs 的维度是 [batch_size, source_seq_len]
        # dec_inputs 的维度是 [batch_size, target_seq_len]
        #-----------------------------------------------------------------
        # 将输入传递给编码器，并获取编码器输出和自注意力权重
        enc_outputs, enc_self_attns = self.encoder(enc_inputs)
        #------------------------- 维度信息 --------------------------------
        # enc_outputs 的维度是 [batch_size, source_len, embedding_dim]
        # enc_self_attns 是一个列表，每个元素的维度是 [batch_size, n_heads, src_seq_len, src_seq_len]
        #-----------------------------------------------------------------
        # 将编码器输出、解码器输入和编码器输入传递给解码器
        # 获取解码器输出、解码器自注意力权重和编码器 - 解码器注意力权重
        dec_outputs, dec_self_attns, dec_enc_attns = self.decoder(dec_inputs, enc_inputs, enc_outputs)
        #------------------------- 维度信息 --------------------------------
        # dec_outputs 的维度是 [batch_size, target_len, embedding_dim]
        # dec_self_attns 是一个列表，每个元素的维度是 [batch_size, n_heads, tgt_seq_len, src_seq_len]
        # dec_enc_attns 是一个列表，每个元素的维度是 [batch_size, n_heads, tgt_seq_len, src_seq_len]
        #-----------------------------------------------------------------
        # 将解码器输出传递给投影层，生成目标词汇表大小的概率分布
        dec_logits = self.projection(dec_outputs)
        #------------------------- 维度信息 --------------------------------
        # dec_logits 的维度是 [batch_size, tgt_seq_len, tgt_vocab_size]
        #-----------------------------------------------------------------
        # 返回逻辑值 ( 原始预测结果 ), 编码器自注意力权重，解码器自注意力权重，解 - 编码器注意力权重
        return dec_logits, enc_self_attns, dec_self_attns, dec_enc_attns

# 定义贪婪解码器函数
def greedy_decoder(model, enc_input, start_symbol):
    # 对输入数据进行编码，并获得编码器输出以及自注意力权重
    enc_outputs, enc_self_attns = model.encoder(enc_input)
    # 初始化解码器输入为全零张量，大小为 (1, 5)，数据类型与 enc_input 一致
    dec_input = torch.zeros(1, 5).type_as(enc_input.data)
    # 设置下一个要解码的符号为开始符号
    next_symbol = start_symbol
    # 循环 5 次，为解码器输入中的每一个位置填充一个符号
    for i in range(0, 5):
        # 将下一个符号放入解码器输入的当前位置
        dec_input[0][i] = next_symbol
        # 运行解码器，获得解码器输出、解码器自注意力权重和编码器 - 解码器注意力权重
        dec_output, _, _ = model.decoder(dec_input, enc_input, enc_outputs)
        # 将解码器输出投影到目标词汇空间
        projected = model.projection(dec_output)
        # 找到具有最高概率的下一个单词
        prob = projected.squeeze(0).max(dim=-1, keepdim=False)[1]
        next_word = prob.data[i]
        # 将找到的下一个单词作为新的符号
        next_symbol = next_word.item()
    # 返回解码器输入，它包含了生成的符号序列
    dec_outputs = dec_input
    return dec_outputs




