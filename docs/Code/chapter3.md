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


## 3.8 Bi-LSTM(Attention)


## 3.9 Transformer


## 3.10 BERT