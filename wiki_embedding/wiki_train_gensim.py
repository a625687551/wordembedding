# -*- coding:utf-8 -*-

import os
import logging
import pickle
import opencc

import jieba_fast as jieba

from gensim.models import word2vec, Word2Vec

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

jieba.enable_parallel(10)


def make_segment_file(file_path):
    print("start seg file")
    jieba.load_userdict("./seg_dict.txt")
    with open(file_path) as f:
        document = f.read()
    d_cut = jieba.cut(document)
    res = " ".join(d_cut)
    with open("./segment_wiki.txt", "w") as f:
        f.write(res)
    print("segment file ok")


def make_vector(model_file):
    print("make vector")
    setences = word2vec.LineSentence("./segment_wiki.txt")
    model = word2vec.Word2Vec(setences, hs=1, min_count=5, window=5, size=300, iter=10, alpha=0.025, min_alpha=0.00001,
                              workers=10)
    model.save(model_file)
    return model


def main():
    model_file = "./wiki.p"
    if not os.path.exists(model_file):
        # make_segment_file("./all")
        model = make_vector(model_file)
    else:
        model = Word2Vec.load(model_file)
    print("model finish")
    print(model.wv.similarity('高数', '线性代数'))
    print(model.wv.most_similar("线性代数"))
    # 加载进来可以继续训练
    # model = Word2Vec.load("word2vec.model")
    # model.train([["hello", "world"]], total_examples=1, epochs=1)


if __name__ == '__main__':
    main()
