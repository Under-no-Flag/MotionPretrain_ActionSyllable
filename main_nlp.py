
# Press the green button in the gutter to run the script.
from dataset.corpus import TranslationCorpus
from model._transformer import Transformer
from torch import nn, optim
from model._transformer import greedy_decoder
if __name__ == '__main__':
    sentences = [
        ['哒哥 喜欢 爬山', 'DaGe likes hiking'],
        ['我 爱 学习 人工智能', 'I love studying AI'],
        ['深度学习 改变 世界', ' DL changed the world'],
        ['自然语言处理 很 强大', 'NLP is powerful'],
        ['神经网络 非常 复杂', 'Neural-networks are complex']]
    # 创建语料库类实例
    corpus = TranslationCorpus(sentences)

    model = Transformer(corpus)  # 创建模型实例
    criterion = nn.CrossEntropyLoss()  # 损失函数
    optimizer = optim.Adam(model.parameters(), lr=0.0001)  # 优化器
    epochs = 10  # 训练轮次
    batch_size=2
    for epoch in range(epochs):  # 训练 100 轮
        optimizer.zero_grad()  # 梯度清零
        enc_inputs, dec_inputs, target_batch = corpus.make_batch(batch_size)  # 创建训练数据
        outputs, _, _, _ = model(enc_inputs, dec_inputs)  # 获取模型输出
        loss = criterion(outputs.view(-1, len(corpus.tgt_vocab)), target_batch.view(-1))  # 计算损失
        if (epoch + 1) % 1 == 0:  # 打印损失
            print(f"Epoch: {epoch + 1:04d} cost = {loss:.6f}")
        loss.backward()  # 反向传播
        optimizer.step()  # 更新参数

    # 用贪婪解码器生成翻译文本
    enc_inputs, dec_inputs, target_batch = corpus.make_batch(batch_size=1, test_batch=True)
    # 使用贪婪解码器生成解码器输入
    greedy_dec_input = greedy_decoder(model, enc_inputs, start_symbol=corpus.tgt_vocab['<sos>'])
    # 将解码器输入转换为单词序列
    greedy_dec_output_words = [corpus.tgt_idx2word[n.item()] for n in greedy_dec_input.squeeze()]
    # 打印编码器输入和贪婪解码器生成的文本
    enc_inputs_words = [corpus.src_idx2word[code.item()] for code in enc_inputs[0]]
    print(enc_inputs_words, '->', greedy_dec_output_words)



