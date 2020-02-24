# 导入相应的包
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(1)


# 准备数据的阶段
def prepare_sequence(seq, to_ix):
    idxs = [to_ix[w] for w in seq]
    return torch.tensor(idxs, dtype=torch.long)


# DET 限定词， NN 名词 V 动词
training_data = [
    ("The dog ate the apple".split(), ["DET", "NN", "V", "DET", "NN"]),
    ("Everybody read that book".split(), ["NN", "V", "DET", "NN"])
]
word_to_ix = {}

for sent, tags in training_data:
    for word in sent:
        if word not in word_to_ix:
            word_to_ix[word] = len(word_to_ix)
print(word_to_ix)
# {'The': 0, 'dog': 1, 'ate': 2, 'the': 3, 'apple': 4, 'Everybody': 5, 'read': 6, 'that': 7, 'book': 8}
tag_to_ix = {"DET": 0, "NN": 1, "V": 2}

# 词向量的维度
EMBEDDING_DIM = 6

# 隐藏层的单元数
HIDDEN_DIM = 6

# 批大小
batch_size = 2


class LSTMTagger(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, vocab_size, tagset_size, batch_size):
        super(LSTMTagger, self).__init__()
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)

        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)

        # The linear layer that maps from hidden state space to tag space
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence):
        embeds = self.word_embeddings(sentence)
        input_tensor = embeds.view(self.batch_size, len(sentence) // self.batch_size, -1)
        lstm_out, _ = self.lstm(input_tensor)
        tag_space = self.hidden2tag(lstm_out)
        tag_scores = F.log_softmax(tag_space, dim=1)
        return tag_scores

    def predict(self, sentence):
        embeds = self.word_embeddings(sentence)
        input_tensor = embeds.view(len(sentence), 1, -1)
        lstm_out, _ = self.lstm(input_tensor)
        tag_space = self.hidden2tag(lstm_out.view(len(sentence), -1))
        tag_scores = F.log_softmax(tag_space, dim=1)
        return tag_scores


def batch_loss_function(batch_size, input_tensor, target_tensor):  # 为了batch实现的损失函数
    temploss_func = torch.zeros(1)
    for i in range(batch_size):
        temploss_func += nn.NLLLoss()(input_tensor[i], target_tensor[i])
    return temploss_func


model = LSTMTagger(EMBEDDING_DIM, HIDDEN_DIM, len(word_to_ix), len(tag_to_ix), batch_size)
optimizer = optim.SGD(model.parameters(), lr=0.1)
# 训练过程
for epoch in range(300):
    for sentence, tags in training_data:
        # 梯度清零
        model.zero_grad()

        # 准备数据
        sentence_in = prepare_sequence(sentence, word_to_ix)
        sentence_in2 = prepare_sequence(sentence, word_to_ix)
        targets = prepare_sequence(tags, tag_to_ix)
        targets2 = prepare_sequence(tags, tag_to_ix)

        # 前向传播
        tag_scores = model(torch.cat([sentence_in, sentence_in2], dim=0))

        # 计算损失
        loss = batch_loss_function(batch_size, tag_scores, torch.cat([targets, targets2], dim=0).view(batch_size, -1))
        # 后向传播
        loss.backward()

        # 更新参数
        optimizer.step()

# 测试过程
with torch.no_grad():
    inputs = prepare_sequence(training_data[0][0], word_to_ix)
    print(inputs)
    tag_scores = model.predict(inputs)
    print(tag_scores.shape)
    print(torch.argmax(tag_scores, dim=1))  # 结果应该为[0,1,2,0,1]
