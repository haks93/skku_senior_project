import numpy as np
from data2numpy import *
from eunjeon import Mecab
import os
import operator
from random import *

rank = 1000
test_set_rate = 0.2
multiple_unit = 1

def refine(filename):
    os.chdir("./demo_data")
    tagger = Mecab()
    nouns_hash = {}

    nouns_hash.clear()
    if filename.endswith(".txt"):

        f = open(filename, "r", encoding='utf-8')
        data = f.read()
        words_pos = tagger.pos(data)
        totalnouns: float = 0

        for word in words_pos:
            noun = word[0]
            category = word[1]
            if (category == "NNG"):
                totalnouns += 1
                if noun in nouns_hash:
                    nouns_hash[noun] = nouns_hash[noun] + 1
                else:
                    nouns_hash[noun] = 1
        f.close()

        for item in nouns_hash:
            nouns_hash[item] = nouns_hash[item] / totalnouns

        sorted_arr = sorted(nouns_hash.items(), key=operator.itemgetter(1), reverse=True)

    os.chdir("../")
    return sorted_arr


def refine_2(rank, multiple_unit, sorted_arr):
    # 각 파일에서 total_top_count.txt 에 있는 단어를 찾고, 그 등장 빈도를 한 줄에 저장. 둘째 줄에는 과목을 저장.
    # test_set_rate에 맞춰서 랜덤으로 train_set, test_set에 나눠서 저장.
    total_rank_hash = {}

    os.chdir("../data/txt/3.nouns_count/")
    with open("total_top_count.txt", 'r', encoding='utf-8') as f:
        lines = f.readlines()
        lines = lines[:rank]
        for line in lines:
            item = line.split(" ")
            total_rank_hash[item[0]] = int(item[1])

    refined_list = [0] * rank

    for item in sorted_arr:
        if item[0] in total_rank_hash:
            refined_list[total_rank_hash[item[0]]] = item[1] * multiple_unit

    np_data = np.array(refined_list, float)
    np_data = np_data.reshape(-1, 1000)

    os.chdir("../../../src")
    return np_data

