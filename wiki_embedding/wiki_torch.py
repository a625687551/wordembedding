# -*- coding:utf-8 -*-


import logging
import jieba_fast as jieba

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(1)
# jieba.enable_parallel(4)
# jieba.load_userdict("seg_dict.txt")

CONTEXT_SIZE = 2
EMBEDDING_DIM = 100
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


def make_cut(model_file):
    with open(model_file, "r", encoding="utf-8") as f:
        doc = f.read()
    d_cut = " ".join(jieba.cut(doc))
    with open("wiki_cut.txt", "w", encoding="utf-8") as f:
        f.write(d_cut)
    return d_cut


def load_cut(cut_file):
    with open(cut_file, "rb") as f:
        d_cut = f.read()
    return d_cut


# raw_text = make_cut("aa").split()
raw_text = load_cut("wiki_cut.txt").split()
print("load finish....")
train_data = [([raw_text[i], raw_text[i + 1]], raw_text[i + 2])
              for i in range(len(raw_text) - 2)]

# build n-gram
# train_data = []
# for i in range(2, len(raw_text)-2):
#     context = [raw_text[i-2], raw_text[i-1], raw_text[i+1], raw_text[i+2]]
#     target = raw_text[i]
#     train_data.append((context, target))
print("n gram finish...")
# one hot
word_to_ix = {}
count = 0
for word in raw_text:
    if word_to_ix.get(word):
        continue
    else:
        word_to_ix[word] = count
        count += 1
print("one hot finish count {}".format(count))


class NGramLanguageModeler(nn.Module):
    def __init__(self, vocab_size, embedding_dim, context_size):
        super(NGramLanguageModeler, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.linear1 = nn.Linear(context_size * embedding_dim, 128)
        self.linear2 = nn.Linear(128, vocab_size)

    def forward(self, inputs):
        embeds = self.embeddings(inputs).view((1, -1))
        out = F.relu(self.linear1(embeds))
        out = self.linear2(out)
        log_probs = F.log_softmax(out, dim=1)
        return log_probs


losses = []
loss_function = nn.NLLLoss()
model = NGramLanguageModeler(vocab_size=count, embedding_dim=EMBEDDING_DIM, context_size=CONTEXT_SIZE)
optimizer = optim.SGD(model.parameters(), lr=0.001)

for epoch in range(5):
    total_loss = 0
    logging.info("epoch is {}".format(epoch))
    for context, target in train_data:
        context_idxs = torch.tensor([word_to_ix[w] for w in context], dtype=torch.long)
        model.zero_grad()
        log_probs = model(context_idxs)
        loss = loss_function(log_probs, torch.tensor([word_to_ix[target]], dtype=torch.long))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    losses.append(total_loss)
print(losses)
