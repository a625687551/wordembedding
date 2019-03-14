# -*- coding:utf-8 -*-

import re
import os
import time
import pickle
import argparse
import urllib

import numpy as np

parse = argparse.ArgumentParser(description="similar words")
parse.add_argument("--path", type=str, default="./vec_saved.p")
parse.add_argument("--word", type=str, default="深度学习")
parse.add_argument("--add_word", type=str, default=None)
parse.add_argument("--add_vocabulary", type=bool, default=False)
parse.add_argument("--top_k", type=int, default=18)
args = parse.parse_args()


class MyEmebededing(object):
    def __init__(self, top_k=18, embed_file="./vec_saved.p"):
        self.embed, self.w_dict = self.get_embed_file(embed_file)
        self.reverse_w_dict = dict(zip(self.w_dict.values(), self.w_dict.keys()))
        self.top_k = top_k

    def get_embed_file(self, path):
        if not os.path.exists(path):
            print("download 300M....")
            filpath, _ = urllib.request.urlretrieve(
                'https://horatio-jsy-1258160473.cos.ap-beijing.myqcloud.com/vec_saved.p', filename=path)
            print("download finish...")
        with open(path, "rb") as f:
            embed, w_dict = pickle.load(f)
        return embed, w_dict

    def get_similar_words(self, word):
        close_word = []
        index = self.w_dict[word[0]]
        word_embed = np.reshape(self.embed[index, :], [1, 300])
        similarity = np.matmul(self.embed, np.transpose(word_embed))
        assert np.shape(similarity) == (len(self.w_dict), 1)
        nearest = (-similarity).argsort(axis=0)[1:self.top_k + 1]
        for k in range(self.top_k):
            close_word.append(self.reverse_w_dict[nearest[k, 0]])
        print("{} is close to : {}".format(word[0], ",".join(close_word)))

    def get_similarity(self, word):
        index = (self.w_dict[word[0]], self.w_dict[word[-1]])
        word_embed = np.reshape(self.embed[[index[0], index[1]], :], [2, 300])
        similarity = np.matmul(word_embed[1, :], np.transpose(word_embed[0, :])) / \
                     np.matmul(word_embed[0, :], np.transpose(word_embed[0, :]))
        print("similariy is {}".format(float(similarity)))

    def get_trends(self, method, word):
        close_word = []
        plus = re.compile(r"\+")
        minus = re.compile(r"\-")
        index = [self.w_dict[word] for word in word]

        if plus.search(args.word) and minus.search(args.word):
            word_embed = self.embed[index[0], :]
            calculate_list = re.split(r"[\u4e00-\u9fa5]{2,}", args.word)
            for i in range(len(calculate_list[1:-1])):
                if calculate_list[1 + i] == "+":
                    word_embed = (word_embed + self.embed[index[1 + i], :]) / 2
                else:
                    word_embed = (word_embed - self.embed[index[1 + i], :]) / 2
            word_embed = np.reshape(word_embed, [1, 300])
        elif method == "-":
            word_embed = (np.reshape(self.embed[index[0], :] - self.embed[index[1], :], [1, 300]))
        else:
            word_embed = (np.reshape(self.embed[index[0], :] + self.embed[index[1], :], [1, 300]))
        similarity = np.matmul(self.embed, np.transpose(word_embed))
        assert np.shape(similarity) == (len(self.w_dict), 1)
        nearest = (-similarity).argsort(axis=0)[1:self.top_k + 1]
        for k in range(self.top_k):
            close_word.append(self.reverse_w_dict[nearest[k, 0]])
        print(" {} is close to :{}".format(args.word, ",".join(close_word)))

    def add_word(self, expression, add_vocabulary):
        w_list = []
        num_list = []
        exp_list = re.split(r"[=*-+]", expression)
        assert len(exp_list) % 2 != 0

        for i in exp_list[1:]:
            if self.w_dict.get(i):
                index_d = self.w_dict[i]
                exist_word_embed = np.reshape(self.embed[index_d, :], [1, 300])
                w_list.append(exist_word_embed)
            elif re.search(r"\d", i):
                num_list.append(i)
            else:
                print('one or more words are not in vocabulary!')
                break
        num_list = np.array(num_list).astype(np.float32)
        assert len(w_list) == len(num_list)
        new_embed = 0
        for i in range(len(num_list)):
            new_embed += num_list[i] * w_list[i]
        self.w_dict[exp_list[0]] = len(self.w_dict)
        self.embed = np.vstack((self.embed, new_embed / len(num_list)))
        self.reverse_w_dict = dict(zip(self.w_dict.values(), self.w_dict.keys()))
        self.get_similar_words(exp_list)
        if add_vocabulary:
            with open(args.path, 'wb') as f:
                pickle.dump((self.embed, self.w_dict), f)
            print('Successfully update vocabulary: %s' % exp_list[0])


def main():
    if args.add_word:
        my_embed = MyEmebededing()
        my_embed.add_word(args.add_word, args.add_vocabulary)
    else:
        word_list = re.split(r'\-|\/|\+', args.word.rstrip())
        search_obj = re.search(r'\-|\/|\+', args.word)
        my_machine = MyEmebededing(args.top_k)
        w_dict = my_machine.w_dict
        contain_words = [word for word in word_list if w_dict.__contains__(word)]

        start = time.time()
        if search_obj is None:
            if w_dict.__contains__(args.word):
                my_machine.get_similar_words(word_list)
            else:
                print('{} is not in vocabulary!'.format(args.word))
        elif len(contain_words) == len(word_list):
            if search_obj.group() == '/':
                my_machine.get_similarity(word_list)
            else:
                my_machine.get_trends(search_obj.group(), word_list)
        else:
            print('%s or %s is not in vocabulary!' % (word_list[0], word_list[-1]))
        print('Inference time:', time.time() - start)


if __name__ == '__main__':
    main()
