# -*- coding:utf-8 -*-

import os
import logging
import pickle

import jieba_fast as jieba
import jieba_fast.analyse

from gensim.models import word2vec

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


def make_segment_file():
    print("seement file start")
    jieba.suggest_freq('沙瑞金', True)
    jieba.suggest_freq('田国富', True)
    jieba.suggest_freq('高育良', True)
    jieba.suggest_freq('侯亮平', True)
    jieba.suggest_freq('钟小艾', True)
    jieba.suggest_freq('陈岩石', True)
    jieba.suggest_freq('欧阳菁', True)
    jieba.suggest_freq('易学习', True)
    jieba.suggest_freq('王大路', True)
    jieba.suggest_freq('蔡成功', True)
    jieba.suggest_freq('孙连城', True)
    jieba.suggest_freq('季昌明', True)
    jieba.suggest_freq('丁义珍', True)
    jieba.suggest_freq('郑西坡', True)
    jieba.suggest_freq('赵东来', True)
    jieba.suggest_freq('高小琴', True)
    jieba.suggest_freq('赵瑞龙', True)
    jieba.suggest_freq('林华华', True)
    jieba.suggest_freq('陆亦可', True)
    jieba.suggest_freq('刘新建', True)
    jieba.suggest_freq('刘庆祝', True)

    with open("./in_the_name_of_people.txt") as f:
        document = f.read()
        d_cut = jieba.cut(document)
        res = " ".join(d_cut)
    with open("./segment_doc.txt", "w") as f:
        f.write(res)
    print("segment file ok")


def make_vector(model_file):
    print("make vector")
    setences = word2vec.LineSentence("./segment_doc.txt")
    model = word2vec.Word2Vec(setences, hs=1, min_count=1, window=3, size=100)
    with open(model_file, "wb") as f:
        pickle.dump(model, f)
    return model


def load_model(model_file):
    with open(model_file, "rb") as f:
        model = pickle.load(f.read())
    return model


def main():
    model_file = "./myword.p"
    if not os.path.exists(model_file):
        make_segment_file()
        model = make_vector(model_file)
    else:
        model = load_model(model_file)
    # 找相似词语
    for key in model.wv.similar_by_word('沙瑞金', topn=100):
        if len(key[0]) == 3:
            print(key[0], key[1])
    # 求相似度
    print(model.wv.similarity('沙瑞金', '高育良'))
    print(model.wv.similarity('李达康', '王大路'))
    # 分类
    print(model.wv.doesnt_match(u"沙瑞金 高育良 李达康 刘庆祝".split()))

if __name__ == '__main__':
    main()