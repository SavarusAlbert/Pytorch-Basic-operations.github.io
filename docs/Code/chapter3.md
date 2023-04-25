# 模型代码部分
## 3.1 NNLM
- 导入工具包
```python
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
```
- 定义函数，将句子分成前 $n-1$ 个单词和第 $n$ 个单词
```python
def make_batch(sentences, word_dict):
    input_batch = []
    target_batch = []

    for sen in sentences:
        word = sen.split()
        input = [word_dict[n] for n in word[:-1]]
        target = word_dict[word[-1]]

        input_batch.append(input)
        target_batch.append(target)
    # 转成long类型才能作为nn.Embedding的输入
    input_batch = torch.LongTensor(input_batch)
    target_batch = torch.LongTensor(target_batch)

    return input_batch, target_batch
```
- 定义模型
```python
class NNLM(nn.Module):
    def __init__(self, n_gram, m, hidden_dim, class_dim):
        super(NNLM, self).__init__()
        self.n_gram = n_gram
        self.m = m
        self.hidden_dim = hidden_dim
        self.class_dim = class_dim
        self.C = nn.Embedding(self.class_dim, m)
        self.H = nn.Linear(self.n_gram * self.m, self.hidden_dim, bias=False)
        self.d = nn.Parameter(torch.ones(self.hidden_dim))
        self.U = nn.Linear(self.hidden_dim, self.class_dim, bias=False)
        self.W = nn.Linear(self.n_gram * self.m, self.class_dim, bias=False)
        self.b = nn.Parameter(torch.ones(self.class_dim))

    def forward(self, X):
        # 输入X的大小为 [batch_size, sen_len]
        X = self.C(X)                                   # 得到 [batch_size, sen_len, m = embedding_dim]
        X = X.view(-1, self.n_gram * self.m)
        tanh = torch.tanh(self.d + self.H(X))           # 得到 [batch_size, hidden_dim]
        output = self.b + self.W(X) + self.U(tanh)      # 得到 [batch_size, class_dim]
        return output
```
- 模型训练函数
```python
def train_NNML(sentences, number_dict, input_batch, target_batch, n_gram, m, hidden_dim, class_dim, num_epoch, lr):
    model = NNLM(n_gram, m, hidden_dim, class_dim)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in tqdm(range(num_epoch)):
        output = model(input_batch)
        loss = criterion(output, target_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 1000 == 0:
            print('Epoch:', '%04d' % (epoch+1), 'cost =', '{:.6f}'.format(loss))

    # 模型预测
    predict = model(input_batch).data.max(1, keepdim=True)[1]
    print([sen.split()[:2] for sen in sentences], '->', [number_dict[n.item()] for n in predict.squeeze()])
```
- 训练举例
```python
if __name__ == '__main__':
    sentences = ['i like dog', 'i love coffee', 'i hate milk']
    word_list = ' '.join(sentences).split()
    word_list = list(set(word_list))
    word_dict = {w: i for i, w in enumerate(word_list)}
    number_dict = {i: w for i, w in enumerate(word_list)}
    class_dim = len(word_dict)

    num_epoch = 5000
    lr = 1e-3
    n_gram = 2
    m = 2
    hidden_dim = 2

    input_batch, target_batch = make_batch(sentences, word_dict)
    train_NNML(sentences, number_dict, input_batch, target_batch, n_gram, m, hidden_dim, class_dim, num_epoch, lr)
```

## 3.2 TextCNN
- 导入工具包
```python
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
```
- 定义模型
```python
class TextCNN(nn.Module):
    def __init__(self, filter_sizes, embedding_size, num_filters, vocab_size, num_classes):
        super().__init__()
        self.filter_sizes = filter_sizes
        self.num_filters_total = num_filters * len(filter_sizes)
        self.emb = nn.Embedding(vocab_size, embedding_size)
        self.W = nn.Linear(self.num_filters_total, num_classes, bias=False)
        self.b = nn.Parameter(torch.ones(num_classes))
        # 卷积的输入通道为1.输出通道为卷积核个数，卷积核大小为filter_sizes*embedding_size
        self.filter_list = nn.ModuleList([nn.Conv2d(1, num_filters, (size, embedding_size)) for size in filter_sizes])

    def forward(self, X):
        # X 的维度为 [batch_size, sequence_length]
        embedded_chars = self.emb(X)                    # 得到 [batch_size, sequence_length, embedding_size]
        embedded_chars = embedded_chars.unsqueeze(1)    # 增加通道 得到 [batch_size, channel(=1), sequence_length, embedding_size]

        pooled_output = []
        # 卷积输出为 [batch_size, num_filters, sequence_length-filter_sizes+1, 1]
        # 最大池化，kernel_size=stride=[sequence_length-filter_sizes+1, 1]，输出大小为 [batch_size, num_filters, 1, 1]
        for i, conv in enumerate(self.filter_list):
            c = F.relu(conv(embedded_chars))
            maxpool = nn.MaxPool2d((sequence_length - self.filter_sizes[i] + 1, 1))
            pooled = maxpool(c)
            pooled_output.append(pooled)

        c_hat = torch.cat(pooled_output, 1)             # 沿通道数concat
        # 展平放入全连接层中
        c_pool_flat = torch.reshape(c_hat, [-1, self.num_filters_total])
        output = self.W(c_pool_flat) + self.b
        return output
```
- 模型训练
```python
def Train_textcnn(input_batch, target_batch, lr, filter_sizes, embedding_size, num_filters, vocab_size, num_classes):
    model = TextCNN(filter_sizes, embedding_size, num_filters, vocab_size, num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # 转成long类型才能作为nn.Embedding的输入
    inputs = torch.LongTensor(input_batch)
    targets = torch.LongTensor(target_batch)
    # 模型训练
    for epoch in tqdm(range(5000)):
        optimizer.zero_grad()
        output = model(inputs)
        loss = criterion(output, targets)
        if (epoch + 1) % 1000 == 0:
            print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.6f}'.format(loss))
        loss.backward()
        optimizer.step()
    # Test 测试样例
    test_text = 'sorry hate you'
    tests = [np.asarray([word_dict[n] for n in test_text.split()])]
    test_batch = torch.LongTensor(tests)
    # Predict 针对测试样例的预测
    predict = model(test_batch).data.max(1, keepdim=True)[1]
    if predict[0][0] == 0:
        print(test_text, "is Bad Mean...")
    else:
        print(test_text, "is Good Mean!!")
```
- 训练举例
```python
if __name__ == '__main__':
    embedding_size = 2
    sequence_length = 3
    num_classes = 2
    filter_sizes = [2, 2, 2]
    num_filters = 3
    lr = 1e-3

    sentences = ["i love you", "he loves me", "she likes baseball", "i hate you", "sorry for that", "this is awful"]
    labels = [1, 1, 1, 0, 0, 0]  # 1 is good, 0 is not good.
    word_list = " ".join(sentences).split()
    word_list = list(set(word_list))
    word_dict = {w: i for i, w in enumerate(word_list)}
    vocab_size = len(word_dict)

    input_batch = [np.asarray([word_dict[n] for n in sen.split()]) for sen in sentences]
    target_batch = [out for out in labels]

    Train_textcnn(input_batch, target_batch, lr, filter_sizes, embedding_size, num_filters, vocab_size, num_classes)
```

## 3.3 TextRNN
- 导入工具包
```python
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
```
- 分割输入和预测
```python
def make_batch(sentences, word_dict, n_class):
    input_batch = []
    target_batch = []

    for sen in sentences:
        word = sen.split()
        input = [word_dict[n] for n in word[:-1]]
        target = word_dict[word[-1]]

        input_batch.append(np.eye(n_class)[input])      # 提取单词的独热编码
        target_batch.append(target)

    return input_batch, target_batch
```
- 定义模型
```python
class TextRNN(nn.Module):
    def __init__(self, n_class, hidden_dim):
        super().__init__()
        self.rnn = nn.RNN(input_size=n_class, hidden_size=hidden_dim)
        self.W = nn.Linear(hidden_dim, n_class, bias=False)
        self.b = nn.Parameter(torch.ones(n_class))

    def forward(self, X, hidden):
        X = X.transpose(0, 1)
        X, hidden = self.rnn(X, hidden)
        X = X[-1]
        outputs = self.W(X) + self.b
        return outputs
```
- 模型训练
```python
def train_TextRNN(n_class, hidden_dim, lr, sentences, word_dict, batch_size, number_dict):
    model = TextRNN(n_class, hidden_dim)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    input_batch, target_batch = make_batch(sentences, word_dict, n_class)
    input_batch = torch.FloatTensor(input_batch)
    target_batch = torch.LongTensor(target_batch)

    for epoch in tqdm(range(5000)):
        optimizer.zero_grad()
        hidden = torch.zeros(1, batch_size, hidden_dim)
        output = model(input_batch, hidden)
        loss = criterion(output, target_batch)
        if (epoch + 1) % 1000 == 0:
            print('Epoch:', '%04d' % (epoch + 1), 'cost = ', '{:.6f}'.format(loss))
        loss.backward()
        optimizer.step()

    # Predict
    hidden = torch.zeros(1, batch_size, hidden_dim)
    predict = model(input_batch, hidden).data.max(1, keepdim=True)[1]
    print([sen.split()[:2] for sen in sentences], '->', [number_dict[n.item()] for n in predict.squeeze()])
```
- 训练举例
```python
if __name__ == '__main__':
    n_step = 2
    hidden_dim = 5
    sentences = ["i like dog", "i love coffee", "i hate milk"]

    word_list = ' '.join(sentences).split()
    word_list = list(set(word_list))
    word_dict = {w: i for i, w in enumerate(word_list)}
    number_dict = {i: w for i, w in enumerate(word_list)}
    n_class = len(word_dict)
    batch_size = len(sentences)
    lr = 1e-3

    train_TextRNN(n_class, hidden_dim, lr, sentences, word_dict, batch_size, number_dict)
```

## 3.4 TextLSTM
- 导入工具包
```python
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
```
- 分割输入和预测
```python
def make_batch(seq_data, word_dict, n_class):
    input_batch = []
    target_batch = []

    for seq in seq_data:
        input = [word_dict[n] for n in seq[:-1]]
        target = word_dict[seq[-1]]

        input_batch.append(np.eye(n_class)[input])      # 提取单词的独热编码
        target_batch.append(target)

    return input_batch, target_batch
```
- 定义模型
```python
class TextLSTM(nn.Module):
    def __init__(self, n_class, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(input_size=n_class, hidden_size=hidden_dim)
        self.W = nn.Linear(hidden_dim, n_class, bias=False)
        self.b = nn.Parameter(torch.ones(n_class))

    def forward(self, X):
        input = X.transpose(0, 1)
        hidden = torch.zeros(1, len(X), self.hidden_dim)
        cell = torch.zeros(1, len(X), self.hidden_dim)

        output, (_, _) = self.lstm(input, (hidden, cell))
        output = output[-1]
        outputs = self.W(output) + self.b
        return outputs
```
- 模型训练
```python
def train_TextLSTM(n_class, hidden_dim, lr, seq_data, word_dict, number_dict):
    model = TextLSTM(n_class, hidden_dim)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    input_batch, target_batch = make_batch(seq_data, word_dict, n_class)
    input_batch = torch.FloatTensor(input_batch)
    target_batch = torch.LongTensor(target_batch)

    for epoch in tqdm(range(1000)):
        optimizer.zero_grad()
        output = model(input_batch)
        loss = criterion(output, target_batch)
        if (epoch + 1) % 100 == 0:
            print('Epoch:', '%04d' % (epoch + 1), 'cost = ', '{:.6f}'.format(loss))
        loss.backward()
        optimizer.step()

    inputs = [sen[:3] for sen in seq_data]
    # Predict
    predict = model(input_batch).data.max(1, keepdim=True)[1]
    print(inputs, '->', [number_dict[n.item()] for n in predict.squeeze()])
```
- 训练举例
```python
if __name__ == '__main__':
    n_step = 3 # number of cells(= number of Step)
    hidden_dim = 128 # number of hidden units in one cell

    char_arr = [c for c in 'abcdefghijklmnopqrstuvwxyz']
    word_dict = {n: i for i, n in enumerate(char_arr)}
    number_dict = {i: w for i, w in enumerate(char_arr)}
    n_class = len(word_dict)  # number of class(=number of vocab)
    lr = 1e-3

    seq_data = ['make', 'need', 'coal', 'word', 'love', 'hate', 'live', 'home', 'hash', 'star']

    train_TextLSTM(n_class, hidden_dim, lr, seq_data, word_dict, number_dict)
```

## 3.5 Bi-LSTM
- 导入工具包
```python
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
```
- 分割输入和预测
```python
def make_batch(sentence, word_dict, n_class, max_len):
    input_batch = []
    target_batch = []

    words = sentence.split()
    for i, word in enumerate(words[:-1]):
        input = [word_dict[n] for n in words[:i+1]]
        input = input + [0]*(max_len - len(input))
        target = word_dict[words[i+1]]
        input_batch.append(np.eye(n_class)[input])
        target_batch.append(target)

    return input_batch, target_batch
```
- 定义模型
```python
class BiLSTM(nn.Module):
    def __init__(self, n_class, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(input_size=n_class, hidden_size=hidden_dim, bidirectional=True)
        self.W = nn.Linear(hidden_dim * 2, n_class, bias=False)
        self.b = nn.Parameter(torch.ones(n_class))

    def forward(self, X):
        input = X.transpose(0, 1)
        hidden = torch.zeros(1*2, len(X), self.hidden_dim)
        cell = torch.zeros(1*2, len(X), self.hidden_dim)

        output, (_, _) = self.lstm(input, (hidden, cell))
        output = output[-1]
        outputs = self.W(output) + self.b
        return outputs
```
- 模型训练
```python
def train_BiLSTM(n_class, hidden_dim, lr, sentence, word_dict, number_dict, max_len):
    model = BiLSTM(n_class, hidden_dim)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    input_batch, target_batch = make_batch(sentence, word_dict, n_class, max_len)
    input_batch = torch.FloatTensor(input_batch)
    target_batch = torch.LongTensor(target_batch)

    for epoch in tqdm(range(10000)):
        optimizer.zero_grad()
        output = model(input_batch)
        loss = criterion(output, target_batch)
        if (epoch + 1) % 1000 == 0:
            print('Epoch:', '%04d' % (epoch + 1), 'cost = ', '{:.6f}'.format(loss))
        loss.backward()
        optimizer.step()

    # Predict
    predict = model(input_batch).data.max(1, keepdim=True)[1]
    print(sentence)
    print([number_dict[n.item()] for n in predict.squeeze()])
```
- 训练举例
```python
if __name__ == '__main__':
    hidden_dim = 5 # number of hidden units in one cell

    sentence = (
        'Lorem ipsum dolor sit amet consectetur adipisicing elit '
        'sed do eiusmod tempor incididunt ut labore et dolore magna '
        'aliqua Ut enim ad minim veniam quis nostrud exercitation'
    )

    word_dict = {w: i for i, w in enumerate(list(set(sentence.split())))}
    number_dict = {i: w for i, w in enumerate(list(set(sentence.split())))}
    n_class = len(word_dict)
    max_len = len(sentence.split())
    lr = 1e-3

    train_BiLSTM(n_class, hidden_dim, lr, sentence, word_dict, number_dict, max_len)
```

## 3.6 Seq2Seq
- 导入工具包
```python
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
```
- 分割输入和预测
```python
def make_batch(seq_data, num_dict, n_class, n_step):
    input_batch, output_batch, target_batch = [], [], []

    for seq in seq_data:
        for i in range(2):
            seq[i] = seq[i] + 'P' * (n_step - len(seq[i]))

        input = [num_dict[n] for n in seq[0]]
        output = [num_dict[n] for n in ('S' + seq[1])]
        target = [num_dict[n] for n in (seq[1] + 'E')]

        input_batch.append(np.eye(n_class)[input])
        output_batch.append(np.eye(n_class)[output])
        target_batch.append(target)

    return torch.FloatTensor(input_batch), torch.FloatTensor(output_batch), torch.LongTensor(target_batch)
def make_testbatch(input_word, num_dict, n_step):
    input_batch, output_batch = [], []

    input_w = input_word + 'P' * (n_step - len(input_word))
    input = [num_dict[n] for n in input_w]
    output = [num_dict[n] for n in 'S' + 'P' * n_step]

    input_batch = np.eye(n_class)[input]
    output_batch = np.eye(n_class)[output]

    return torch.FloatTensor(input_batch).unsqueeze(0), torch.FloatTensor(output_batch).unsqueeze(0)
```
- 定义模型
```python
class seq2seq(nn.Module):
    def __init__(self, n_class, hidden_dim):
        super().__init__()
        self.enc = nn.RNN(input_size=n_class, hidden_size=hidden_dim, dropout=0.5)
        self.dec = nn.RNN(input_size=n_class, hidden_size=hidden_dim, dropout=0.5)
        self.fc = nn.Linear(hidden_dim, n_class)

    def forward(self, enc_input, enc_hidden, dec_input):
        enc_input = enc_input.transpose(0, 1)
        dec_input = dec_input.transpose(0, 1)

        _, enc_states = self.enc(enc_input, enc_hidden)
        dec_outputs, _ = self.dec(dec_input, enc_states)
        outputs = self.fc(dec_outputs)
        return outputs
```
- 模型训练
```python
def train_seq2seq(n_class, hidden_dim, lr, seq_data, num_dict, n_step, batch_size):
    model = seq2seq(n_class, hidden_dim)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    input_batch, output_batch, target_batch = make_batch(seq_data, num_dict, n_class, n_step)

    for epoch in tqdm(range(5000)):
        hidden = torch.zeros(1, batch_size, hidden_dim)
        optimizer.zero_grad()
        output = model(input_batch, hidden, output_batch)
        output = output.transpose(0, 1)
        loss = 0
        for i in range(len(target_batch)):
            loss += criterion(output[i], target_batch[i])
        if (epoch + 1) % 1000 == 0:
            print('Epoch:', '%04d' % (epoch + 1), 'cost = ', '{:.6f}'.format(loss))
        loss.backward()
        optimizer.step()

    # Text
    def translate(word):
        input_batch, output_batch = make_testbatch(word, num_dict, n_step)

        # make hidden shape [num_layers * num_directions, batch_size, n_hidden]
        hidden = torch.zeros(1, 1, hidden_dim)
        output = model(input_batch, hidden, output_batch)
        # output : [max_len+1(=6), batch_size(=1), n_class]

        predict = output.data.max(2, keepdim=True)[1]  # select n_class dimension
        decoded = [char_arr[i] for i in predict]
        end = decoded.index('E')
        translated = ''.join(decoded[:end])

        return translated.replace('P', '')

    print('test')
    print('man ->', translate('man'))
    print('mans ->', translate('mans'))
    print('king ->', translate('king'))
    print('black ->', translate('black'))
    print('upp ->', translate('upp'))
```
- 训练举例
```python
if __name__ == '__main__':
    n_step = 5
    hidden_dim = 128

    char_arr = [c for c in 'SEPabcdefghijklmnopqrstuvwxyz']
    num_dict = {n: i for i, n in enumerate(char_arr)}
    seq_data = [['man', 'women'], ['black', 'white'], ['king', 'queen'], ['girl', 'boy'], ['up', 'down'],
                ['high', 'low']]

    n_class = len(num_dict)
    batch_size = len(seq_data)
    lr = 1e-3

    train_seq2seq(n_class, hidden_dim, lr, seq_data, num_dict, n_step, batch_size)
```

## 3.7 Seq2Seq(Attention)
- 导入工具包
```python
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import matplotlib.pyplot as plt
```
- 分割输入和预测
```python
def make_batch(sentences, word_dict, n_class):
    input_batch = [np.eye(n_class)[[word_dict[n] for n in sentences[0].split()]]]
    output_batch = [np.eye(n_class)[[word_dict[n] for n in sentences[1].split()]]]
    target_batch = [[word_dict[n] for n in sentences[2].split()]]

    return torch.FloatTensor(input_batch), torch.FloatTensor(output_batch), torch.LongTensor(target_batch)
```
- 定义模型
```python
class Attention(nn.Module):
    def __init__(self, n_class, n_hidden):
        super().__init__()
        self.n_class = n_class
        self.enc = nn.RNN(input_size=n_class, hidden_size=n_hidden, dropout=0.5)
        self.dec = nn.RNN(input_size=n_class, hidden_size=n_hidden, dropout=0.5)

        self.attn = nn.Linear(n_hidden, n_hidden)
        self.out = nn.Linear(n_hidden * 2, n_class)

    def forward(self, enc_input, hidden, dec_input):
        # enc_input : [n_step, batch_size, n_class]
        # dec_input : [n_step, batch_size, n_class]
        # hidden : [1, batch_size, n_hidden]
        enc_input = enc_input.transpose(0, 1)
        dec_input = dec_input.transpose(0, 1)

        # enc_outputs : [n_step, batch_size, num_directions(=1) * n_hidden]
        # enc_hidden : [num_layers(=1) * num_directions(=1), batch_size, n_hidden]
        enc_outputs, enc_hidden = self.enc(enc_input, hidden)

        trained_attn = []
        hidden = enc_hidden
        n_step = len(dec_input)
        outputs = torch.empty([n_step, 1, self.n_class])

        for i in range(n_step):
            # dec_input : [n_step, batch_size, n_class]
            # dec_input[i] : [batch_size, n_class]
            # dec_input[i].unsqueeze(0) : [1, batch_size, n_class]
            # dec_output : [1, batch_size, num_directions(=1) * n_hidden]
            # hidden : [num_layer(=1) * num_directions(=1), batch_size, n_hidden]
            dec_output, hidden = self.dec(dec_input[i].unsqueeze(0), hidden)
            attn_weights = self.get_att_weight(enc_outputs, dec_output)
            # attn_weights : [1, 1, n_step]
            # .squeeze() 删除所有shape为1的维度
            trained_attn.append(attn_weights.squeeze().data.numpy())

            # [1, 1, n_step] * [batch_size(=1), n_step, num_directions(=1) * n_hidden] = [1, 1, num_directions(=1) * n_hidden]
            context = attn_weights.bmm(enc_outputs.transpose(0, 1))
            # dec_output : [batch_size(=1), num_directions(=1) * n_hidden]
            dec_output = dec_output.squeeze(0)
            # context : [1, num_directions(=1) * n_hidden]
            context = context.squeeze(1)
            # outputs : [n_step, 1, n_class]
            outputs[i] = self.out(torch.cat((dec_output, context), 1))

        return outputs.transpose(0, 1).squeeze(0), trained_attn
    
    def get_att_weight(self, enc_outputs, dec_output):
        n_step = len(enc_outputs)
        attn_scores = torch.zeros(n_step)
        # enc_outputs : [n_step, batch_size, num_directions(=1) * n_hidden]
        # enc_outputs[i] : [batch_size, num_directions(=1) * n_hidden]
        # self.attn(enc_outputs[i]) : [batch_size, n_hidden]
        # dec_output : [1, batch_size, num_directions(=1) * n_hidden]
        for i in range(n_step):
            attn_scores[i] =  torch.dot(self.attn(enc_outputs[i]).view(-1), dec_output.view(-1))
        return F.softmax(attn_scores).view(1, 1, -1)
```
- 模型训练
```python
def train_Attention(n_class, n_hidden, sentences, word_dict, hidden, number_dict):
    model = Attention(n_class, n_hidden)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    input_batch, output_batch, target_batch = make_batch(sentences, word_dict, n_class)
    # Train
    for epoch in tqdm(range(2000)):
        optimizer.zero_grad()
        output, _ = model(input_batch, hidden, output_batch)
        loss = criterion(output, target_batch.squeeze(0))
        if (epoch + 1) % 400 == 0:
            print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.6f}'.format(loss))
        loss.backward()
        optimizer.step()

    # Test
    test_batch = [np.eye(n_class)[[word_dict[n] for n in 'SPPPP']]]
    test_batch = torch.FloatTensor(test_batch)
    predict, trained_attn = model(input_batch, hidden, test_batch)
    predict = predict.data.max(1, keepdim=True)[1]
    print(sentences[0], '->', [number_dict[n.item()] for n in predict.squeeze()])

    # Show Attention
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(1, 1, 1)
    ax.matshow(trained_attn, cmap='viridis')
    ax.set_xticklabels([''] + sentences[0].split(), fontdict={'fontsize': 14})
    ax.set_yticklabels([''] + sentences[2].split(), fontdict={'fontsize': 14})
    plt.show()
```
- 训练举例
```python
if __name__ == '__main__':
    n_step = 5 # number of cells(= number of Step)
    n_hidden = 128 # number of hidden units in one cell

    sentences = ['ich mochte ein bier P', 'S i want a beer', 'i want a beer E']

    word_list = " ".join(sentences).split()
    word_list = list(set(word_list))
    word_dict = {w: i for i, w in enumerate(word_list)}
    number_dict = {i: w for i, w in enumerate(word_list)}
    n_class = len(word_dict)  # vocab list
    lr = 1e-3
    # hidden : [num_layers(=1) * num_directions(=1), batch_size, n_hidden]
    hidden = torch.zeros(1, 1, n_hidden)
    train_Attention(n_class, n_hidden, sentences, word_dict, hidden, number_dict)
```

## 3.8 Bi-LSTM(Attention)
- 导入工具包
```python
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import matplotlib.pyplot as plt
```
- 定义模型
```python
class BiLSTM_Attention(nn.Module):
    def __init__(self, vocab_size, embedding_dim, n_class, n_hidden):
        super().__init__()
        self.n_hidden = n_hidden
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, n_hidden, bidirectional=True)
        self.out = nn.Linear(n_hidden * 2, n_class)

    def forward(self, X):
        batch_size = len(X)
        input = self.embedding(X)
        input = input.permute(1, 0, 2)
        hidden_state = torch.zeros(1*2, batch_size, self.n_hidden)
        cell_state = torch.zeros(1*2, batch_size, self.n_hidden)

        output, (final_hidden_state, final_cell_state) = self.lstm(input, (hidden_state, cell_state))
        output = output.permute(1, 0, 2)
        attn_output, attention = self.attention_net(output, final_hidden_state)
        return self.out(attn_output), attention

    def attention_net(self, lstm_output, final_state):
        hidden = final_state.view(-1, self.n_hidden * 2, 1)
        attn_weights = torch.bmm(lstm_output, hidden).squeeze(2)
        soft_attn_weights = F.softmax(attn_weights, 1)
        context = torch.bmm(lstm_output.transpose(1, 2), soft_attn_weights.unsqueeze(2)).squeeze(2)
        return context, soft_attn_weights.data.numpy()
```
- 模型训练
```python
def train_Attention(n_class, n_hidden, sentences, word_dict, labels, vocab_size, embedding_dim):
    model = BiLSTM_Attention(vocab_size, embedding_dim, n_class, n_hidden)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    inputs = torch.LongTensor([np.asarray([word_dict[n] for n in sen.split()]) for sen in sentences])
    targets = torch.LongTensor([out for out in labels])

    for epoch in tqdm(range(5000)):
        optimizer.zero_grad()
        output, attention = model(inputs)
        loss = criterion(output, targets)
        if (epoch + 1) % 1000 == 0:
            print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.6f}'.format(loss))
        loss.backward()
        optimizer.step()

    # Test
    test_text = 'sorry hate you'
    tests = [np.asarray([word_dict[n] for n in test_text.split()])]
    test_batch = torch.LongTensor(tests)

    # Predict
    predict, _ = model(test_batch)
    predict = predict.data.max(1, keepdim=True)[1]
    if predict[0][0] == 0:
        print(test_text, "is Bad Mean...")
    else:
        print(test_text, "is Good Mean!!")

    fig = plt.figure(figsize=(6, 3))  # [batch_size, n_step]
    ax = fig.add_subplot(1, 1, 1)
    ax.matshow(attention, cmap='viridis')
    ax.set_xticklabels([''] + ['first_word', 'second_word', 'third_word'], fontdict={'fontsize': 14}, rotation=90)
    ax.set_yticklabels([''] + ['batch_1', 'batch_2', 'batch_3', 'batch_4', 'batch_5', 'batch_6'],
                       fontdict={'fontsize': 14})
    plt.show()
```
- 训练举例
```python
if __name__ == '__main__':
    embedding_dim = 2  # embedding size
    n_hidden = 5  # number of hidden units in one cell
    n_class = 2  # 0 or 1

    # 3 words sentences (=sequence_length is 3)
    sentences = ["i love you", "he loves me", "she likes baseball", "i hate you", "sorry for that", "this is awful"]
    labels = [1, 1, 1, 0, 0, 0]  # 1 is good, 0 is not good.

    word_list = " ".join(sentences).split()
    word_list = list(set(word_list))
    word_dict = {w: i for i, w in enumerate(word_list)}
    vocab_size = len(word_dict)
    lr = 1e-3

    train_Attention(n_class, n_hidden, sentences, word_dict, labels, vocab_size, embedding_dim)
```

## 3.9 Transformer
- 模型结构图

![](Transformer.png ':size=80%')
- 导入工具包
```python
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
```
- 分割数据集
```python
def make_batch(sentences):
    input_batch = [[src_vocab[n] for n in sentences[0].split()]]
    output_batch = [[tgt_vocab[n] for n in sentences[1].split()]]
    target_batch = [[tgt_vocab[n] for n in sentences[2].split()]]
    return torch.LongTensor(input_batch), torch.LongTensor(output_batch), torch.LongTensor(target_batch)
```
- 正弦位置编码
```python
def get_sinusoid_encoding_table(n_position, d_model):
    def cal_angle(position, hid_idx):
        return position / np.power(10000, 2 * (hid_idx // 2) / d_model)
    def get_posi_angle_vec(position):
        return [cal_angle(position, hid_j) for hid_j in range(d_model)]

    sinusoid_table = np.array([get_posi_angle_vec(pos_i) for pos_i in range(n_position)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])
    return torch.FloatTensor(sinusoid_table)
```
- 掩码函数
```python
# padding 0 (句子补0部分) 进行mask，形状与q、k一致
def get_attn_pad_mask(seq_q, seq_k):
    batch_size, len_q = seq_q.size()
    batch_size, len_k = seq_k.size()
    pad_attn_mask = seq_k.data.eq(0).unsqueeze(1)
    return pad_attn_mask.expand(batch_size, len_q, len_k)
# 位置编码，上三角矩阵mask需要预测的词后的所有词
def get_attn_subsequent_mask(seq):
    attn_shape = [seq.size(0), seq.size(1), seq.size(1)]
    subsequent_mask = np.triu(np.ones(attn_shape), k=1)
    subsequent_mask = torch.from_numpy(subsequent_mask).byte()
    return subsequent_mask
```
- 点积自注意力
```python
class ScaleDotProductAttention(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, Q, K, V, attn_mask):
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(d_k)
        scores.masked_fill_(attn_mask, -1e9)
        attn = nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attn, V)
        return context, attn
```
- 多头自注意力
```python
class MultiHeadAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.W_Q = nn.Linear(d_model, d_k * n_heads)
        self.W_K = nn.Linear(d_model, d_k * n_heads)
        self.W_V = nn.Linear(d_model, d_k * n_heads)
        self.linear = nn.Linear(n_heads * d_v, d_model)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, Q, K, V, attn_mask):
        residual, batch_size = Q, Q.size(0)
        q_s = self.W_Q(Q).view(batch_size, -1, n_heads, d_k).transpose(1, 2)
        k_s = self.W_K(K).view(batch_size, -1, n_heads, d_k).transpose(1, 2)
        v_s = self.W_V(V).view(batch_size, -1, n_heads, d_v).transpose(1, 2)
        attn_mask = attn_mask.unsqueeze(1).repeat(1, n_heads, 1, 1)

        context, attn = ScaleDotProductAttention()(q_s, k_s, v_s, attn_mask)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, n_heads * d_v)
        output = self.linear(context)
        return self.layer_norm(output + residual), attn
```
- 前馈网络
```python
class PoswiseFeedForwardNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, inputs):
        residual = inputs
        output = nn.ReLU()(self.conv1(inputs.transpose(1, 2)))
        output = self.conv2(output).transpose(1, 2)
        return self.layer_norm(output + residual)
```
- 编码层
```python
class EncoderLayer(nn.Module):
    def __init__(self):
        super(EncoderLayer, self).__init__()
        self.enc_self_attn = MultiHeadAttention()
        self.pos_ffn = PoswiseFeedForwardNet()

    def forward(self, enc_inputs, enc_self_attn_mask):
        enc_outputs, attn = self.enc_self_attn(enc_inputs, enc_inputs, enc_inputs, enc_self_attn_mask)
        enc_outputs = self.pos_ffn(enc_outputs)
        return enc_outputs, attn
```
- 解码层
```python
class DecoderLayer(nn.Module):
    def __init__(self):
        super(DecoderLayer, self).__init__()
        self.dec_self_attn = MultiHeadAttention()
        self.dec_enc_attn = MultiHeadAttention()
        self.pos_ffn = PoswiseFeedForwardNet()

    def forward(self, dec_inputs, enc_outputs, dec_self_attn_mask, dec_enc_attn_mask):
        dec_outputs, dec_self_attn = self.dec_self_attn(dec_inputs, dec_inputs, dec_inputs, dec_self_attn_mask)
        dec_outputs, dec_enc_attn = self.dec_self_attn(dec_outputs, dec_outputs, dec_outputs, dec_enc_attn_mask)
        dec_outputs = self.pos_ffn(dec_outputs)
        return dec_outputs, dec_self_attn, dec_enc_attn
```
- 编码器
```python
class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.src_emb = nn.Embedding(src_vocab_size, d_model)
        self.pos_emb = nn.Embedding.from_pretrained(get_sinusoid_encoding_table(src_len+1, d_model), freeze=True)
        self.layers = nn.ModuleList([EncoderLayer() for _ in range(n_layers)])

    def forward(self, enc_inputs):
        enc_outputs = self.src_emb(enc_inputs) + self.pos_emb(torch.LongTensor([[1,2,3,4,0]]))
        enc_self_attn_mask = get_attn_pad_mask(enc_inputs, enc_inputs)
        enc_self_attns = []
        for layer in self.layers:
            enc_outputs, enc_self_attn = layer(enc_outputs, enc_self_attn_mask)
            enc_self_attns.append(enc_self_attn)
        return  enc_outputs, enc_self_attns
```
- 解码器
```python
class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.tgt_emb = nn.Embedding(tgt_vocab_size, d_model)
        self.pos_emb = nn.Embedding.from_pretrained(get_sinusoid_encoding_table(tgt_len+1, d_model), freeze=True)
        self.layers = nn.ModuleList([DecoderLayer() for _ in range(n_layers)])

    def forward(self, dec_inputs, enc_inputs, enc_outputs):
        dec_outputs = self.tgt_emb(dec_inputs) + self.pos_emb(torch.LongTensor([[5,1,2,3,4]]))
        dec_self_attn_pad_mask = get_attn_pad_mask(dec_inputs, dec_inputs)
        dec_self_attn_subsequent_mask = get_attn_subsequent_mask(dec_inputs)
        dec_self_attn_mask = torch.gt((dec_self_attn_pad_mask + dec_self_attn_subsequent_mask), 0)
        dec_enc_attn_mask = get_attn_pad_mask(dec_inputs, enc_inputs)

        dec_self_attns, dec_enc_attns = [], []
        for layer in self.layers:
            dec_outputs, dec_self_attn, dec_enc_attn = layer(dec_outputs, enc_outputs, dec_self_attn_mask, dec_enc_attn_mask)
            dec_self_attns.append(dec_self_attn)
            dec_enc_attns.append(dec_enc_attn)
        return dec_outputs, dec_self_attns, dec_enc_attns
```
- Transformer
```python
class Transformer(nn.Module):
    def __init__(self):
        super(Transformer, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.projection = nn.Linear(d_model, tgt_vocab_size, bias=False)

    def forward(self, enc_inputs, dec_inputs):
        enc_outputs, enc_self_attns = self.encoder(enc_inputs)
        dec_outputs, dec_self_attns, dec_enc_attns = self.decoder(dec_inputs, enc_inputs, enc_outputs)
        dec_logits = self.projection(dec_outputs)
        return dec_logits.view(-1, dec_logits.size(-1)), enc_self_attns, dec_self_attns, dec_enc_attns
```
- 绘图
```python
def showgraph(attn):
    attn = attn[-1].squeeze(0)[0]
    attn = attn.squeeze(0).data.numpy()
    fig = plt.figure(figsize=(n_heads, n_heads))
    ax = fig.add_subplot(1, 1, 1)
    ax.matshow(attn, cmap='viridis')
    ax.set_xticklabels([''] + sentences[0].split(), fontdict={'fontsize': 14}, rotation=90)
    ax.set_yticklabels([''] + sentences[2].split(), fontdict={'fontsize': 14})
    plt.show()
```
- 模型训练
```python
def train_Transformer():
    model = Transformer()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    enc_inputs, dec_inputs, target_batch = make_batch(sentences)

    for epoch in range(20):
        optimizer.zero_grad()
        outputs, enc_self_attns, dec_self_attns, dec_enc_attns = model(enc_inputs, dec_inputs)
        loss = criterion(outputs, target_batch.contiguous().view(-1))
        print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.6f}'.format(loss))
        loss.backward()
        optimizer.step()

    # Test
    predict, _, _, _ = model(enc_inputs, dec_inputs)
    predict = predict.data.max(1, keepdim=True)[1]
    print(sentences[0], '->', [number_dict[n.item()] for n in predict.squeeze()])

    print('first head of last state enc_self_attns')
    showgraph(enc_self_attns)

    print('first head of last state dec_self_attns')
    showgraph(dec_self_attns)

    print('first head of last state dec_enc_attns')
    showgraph(dec_enc_attns)
```
- 训练举例
```python
if __name__ == '__main__':
    sentences = ['ich mochte ein bier P', 'S i want a beer', 'i want a beer E']

    # Transformer Parameters
    # Padding Should be Zero
    src_vocab = {'P': 0, 'ich': 1, 'mochte': 2, 'ein': 3, 'bier': 4}
    src_vocab_size = len(src_vocab)

    tgt_vocab = {'P': 0, 'i': 1, 'want': 2, 'a': 3, 'beer': 4, 'S': 5, 'E': 6}
    number_dict = {i: w for i, w in enumerate(tgt_vocab)}
    tgt_vocab_size = len(tgt_vocab)

    src_len = 5  # length of source
    tgt_len = 5  # length of target

    d_model = 512  # Embedding Size
    d_ff = 2048  # FeedForward dimension
    d_k = d_v = 64  # dimension of K(=Q), V
    n_layers = 6  # number of Encoder of Decoder Layer
    n_heads = 8  # number of heads in Multi-Head Attention
    lr = 1e-3

    train_Transformer()
```

## 3.10 BERT
- 导入工具包
```python
import math
import re
from random import *
import numpy as np
import torch
import torch.nn as nn
```
- 分割数据集
```python
def make_batch():
    batch = []
    positive = negative = 0
    while positive != batch_size/2 or negative != batch_size/2:
        tokens_a_index, tokens_b_index = randrange(len(sentences)), randrange(len(sentences))
        token_a, token_b = token_list[tokens_a_index], token_list[tokens_b_index]
        input_ids = [word_dict['[CLS]']] + token_a + [word_dict['[SEP]']] + token_b + [word_dict['[SEP]']]
        segment_ids = [0] * (1 + len(token_a) + 1) + [1] * (len(token_b) + 1)

        n_pred = min(max_pred, max(1, int(round(len(input_ids) * 0.15))))
        cand_maked_pos = [i for i, token in enumerate(input_ids)
                          if token != word_dict['[CLS]'] and token != word_dict['[SEP]']]
        shuffle(cand_maked_pos)
        masked_tokens, masked_pos = [], []
        for pos in cand_maked_pos[:n_pred]:
            masked_pos.append(pos)
            masked_tokens.append(input_ids[pos])
            if random() < 0.8:
                input_ids[pos] = word_dict['[MASK]']
            elif random() < 0.5:
                index = randint(0, vocab_size-1)
                input_ids[pos] = word_dict[number_dict[index]]

        n_pad = maxlen - len(input_ids)
        input_ids.extend([0] * n_pad)
        segment_ids.extend([0] * n_pad)

        if max_pred > n_pred:
            n_pad = max_pred - n_pred
            masked_tokens.extend([0] * n_pad)
            masked_pos.extend([0] * n_pad)

        if tokens_a_index + 1 == tokens_b_index and positive < batch_size/2:
            batch.append([input_ids, segment_ids, masked_tokens, masked_pos, True])
            positive += 1
        elif tokens_a_index + 1 != tokens_b_index and negative < batch_size/2:
            batch.append([input_ids, segment_ids, masked_tokens, masked_pos, False])
            negative += 1
    return batch
```
- 掩码函数
```python
def get_attn_pad_mask(seq_q, seq_k):
    batch_size, len_q = seq_q.size()
    batch_size, len_k = seq_k.size()
    pad_attn_mask = seq_k.data.eq(0).unsqueeze(1)
    return pad_attn_mask.expand(batch_size, len_q, len_k)
```
- `gelu` 激活函数
```python
def gelu(x):
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))
```
- 嵌入层
```python
class Embedding(nn.Module):
    def __init__(self):
        super(Embedding, self).__init__()
        self.tok_embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(maxlen, d_model)
        self.seg_embed = nn.Embedding(n_segments, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, seg):
        seq_len = x.size(1)
        pos = torch.arange(seq_len, dtype=torch.long)
        pos = pos.unsqueeze(0).expand_as(x)
        embedding = self.tok_embed(x) + self.pos_embed(pos) + self.seg_embed(seg)
        return self.norm(embedding)
```
- 标量点积自注意力
```python
class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()

    def forward(self, Q, K, V, attn_mask):
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(d_k)
        scores.masked_fill_(attn_mask, -1e9)
        attn = nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attn, V)
        return context, attn
```
- 多头自注意力
```python
class MultiHeadAttention(nn.Module):
    def __init__(self):
        super(MultiHeadAttention, self).__init__()
        self.W_Q = nn.Linear(d_model, d_k * n_heads)
        self.W_K = nn.Linear(d_model, d_k * n_heads)
        self.W_V = nn.Linear(d_model, d_v * n_heads)

    def forward(self, Q, K, V, attn_mask):
        residual, batch_size = Q, Q.size(0)
        q_s = self.W_Q(Q).view(batch_size, -1, n_heads, d_k).transpose(1, 2)
        k_s = self.W_K(K).view(batch_size, -1, n_heads, d_k).transpose(1, 2)
        v_s = self.W_V(V).view(batch_size, -1, n_heads, d_v).transpose(1, 2)

        attn_mask = attn_mask.unsqueeze(1).repeat(1, n_heads, 1, 1)

        context, attn = ScaledDotProductAttention()(q_s, k_s, v_s, attn_mask)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, n_heads * d_v)
        output = nn.Linear(n_heads * d_v, d_model)(context)
        return nn.LayerNorm(d_model)(output + residual), attn
```
- 前馈网络
```python
class PoswiseFeedForwardNet(nn.Module):
    def __init__(self):
        super(PoswiseFeedForwardNet, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        return self.fc2(gelu(self.fc1(x)))
```
- 编码器
```python
class EncoderLayer(nn.Module):
    def __init__(self):
        super(EncoderLayer, self).__init__()
        self.enc_self_attn = MultiHeadAttention()
        self.pos_ffn = PoswiseFeedForwardNet()

    def forward(self, enc_inputs, enc_self_attn_mask):
        enc_outputs, attn = self.enc_self_attn(enc_inputs, enc_inputs, enc_inputs, enc_self_attn_mask)
        enc_outputs = self.pos_ffn(enc_outputs)
        return enc_outputs, attn
```
- BERT
```python
class BERT(nn.Module):
    def __init__(self):
        super(BERT, self).__init__()
        self.embedding = Embedding()
        self.layers = nn.ModuleList([EncoderLayer() for _ in range(n_layers)])
        self.fc = nn.Linear(d_model, d_model)
        self.activ1 = nn.Tanh()
        self.linear = nn.Linear(d_model, d_model)
        self.activ2 = gelu
        self.norm = nn.LayerNorm(d_model)
        self.classifier = nn.Linear(d_model, 2)
        embed_weight = self.embedding.tok_embed.weight
        n_vocab, n_dim = embed_weight.size()
        self.decoder = nn.Linear(n_dim, n_vocab, bias=False)
        self.decoder.weight = embed_weight
        self.decoder_bias = nn.Parameter(torch.zeros(n_vocab))

    def forward(self, input_ids, segment_ids, masked_pos):
        output = self.embedding(input_ids, segment_ids)
        enc_self_attn_mask = get_attn_pad_mask(input_ids, input_ids)
        for layer in self.layers:
            output, enc_self_attn = layer(output, enc_self_attn_mask)
        h_pooled = self.activ1(self.fc(output[:, 0]))
        logits_clsf = self.classifier(h_pooled)

        masked_pos = masked_pos[:, :, None].expand(-1, -1, output.size(-1))
        h_masked = torch.gather(output, 1, masked_pos)
        h_masked = self.norm(self.activ2(self.linear(h_masked)))
        logits_lm = self.decoder(h_masked) + self.decoder_bias
        return logits_lm, logits_clsf
```
- 模型训练
```python
def train_BERT():
    model = BERT()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    batch = make_batch()
    input_ids, segment_ids, masked_tokens, masked_pos, isNext = map(torch.LongTensor, zip(*batch))

    for epoch in range(100):
        optimizer.zero_grad()
        logits_lm, logits_clsf = model(input_ids, segment_ids, masked_pos)
        loss_lm = criterion(logits_lm.transpose(1, 2), masked_tokens)
        loss_lm = (loss_lm.float()).mean()
        loss_clsf = criterion(logits_clsf, isNext)
        loss = loss_lm + loss_clsf
        if (epoch + 1) % 10 == 0:
            print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.6f}'.format(loss))
        loss.backward()
        optimizer.step()

    # Predict mask tokens ans isNext
    input_ids, segment_ids, masked_tokens, masked_pos, isNext = map(torch.LongTensor, zip(batch[0]))
    print(text)
    print([number_dict[w.item()] for w in input_ids[0] if number_dict[w.item()] != '[PAD]'])

    logits_lm, logits_clsf = model(input_ids, segment_ids, masked_pos)
    logits_lm = logits_lm.data.max(2)[1][0].data.numpy()
    print('masked tokens list : ', [pos.item() for pos in masked_tokens[0] if pos.item() != 0])
    print('predict masked tokens list : ', [pos for pos in logits_lm if pos != 0])

    logits_clsf = logits_clsf.data.max(1)[1].data.numpy()[0]
    print('isNext : ', True if isNext else False)
    print('predict isNext : ', True if logits_clsf else False)
```
- 训练举例
```python
if __name__ == '__main__':
    # BERT Parameters
    maxlen = 30 # maximum of length
    batch_size = 6
    max_pred = 5  # max tokens of prediction
    n_layers = 6 # number of Encoder of Encoder Layer
    n_heads = 12 # number of heads in Multi-Head Attention
    d_model = 768 # Embedding Size
    d_ff = 768 * 4  # 4*d_model, FeedForward dimension
    d_k = d_v = 64  # dimension of K(=Q), V
    n_segments = 2
    lr = 1e-3

    text = (
        'Hello, how are you? I am Romeo.\n'
        'Hello, Romeo My name is Juliet. Nice to meet you.\n'
        'Nice meet you too. How are you today?\n'
        'Great. My baseball team won the competition.\n'
        'Oh Congratulations, Juliet\n'
        'Thanks you Romeo'
    )
    sentences = re.sub("[.,!?\\-]", '', text.lower()).split('\n')  # filter '.', ',', '?', '!'
    word_list = list(set(" ".join(sentences).split()))
    word_dict = {'[PAD]': 0, '[CLS]': 1, '[SEP]': 2, '[MASK]': 3}
    for i, w in enumerate(word_list):
        word_dict[w] = i + 4
    number_dict = {i: w for i, w in enumerate(word_dict)}
    vocab_size = len(word_dict)

    token_list = list()
    for sentence in sentences:
        arr = [word_dict[s] for s in sentence.split()]
        token_list.append(arr)

    train_BERT()
```