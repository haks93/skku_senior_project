from eunjeon import Mecab
import os
import operator
from random import *


def save_each_txt_rate(subject_index):
    # 각 파일에서 단어의 등장 비율을 계산
    # 0: eng, 1: kor,  2: math, 3: science
    subj = ["english", "korean", "mathematics", "science"]

    subject = subj[subject_index]

    os.chdir("../data/txt/2.txt_files/"+subject)
    tagger = Mecab()
    nouns_hash = {}

    for subsection in  os.listdir(os.getcwd()):
        os.chdir("./" + subsection)

        for filename in os.listdir(os.getcwd()):
            nouns_hash.clear()
            if filename.endswith(".txt"):
                # print(os.getcwd()+"\\"+filename)

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

                # 여기에서 각 파일별로 txt파일 쓸 수 있게 수정. 경로 설정이 번거로울듯.
                sorted_arr = sorted(nouns_hash.items(), key=operator.itemgetter(1), reverse=True)

                os.chdir("../../../4.nouns_rate/" + subject + "/" + subsection)

                f = open(filename, "w", encoding='utf-8')

                for item in sorted_arr:
                    f.write(item[0] + " " + str(item[1]/totalnouns)[:8] + "\n")
                f.close()

                os.chdir("../../../2.txt_files/" + subject + "/" + subsection)

                # 경로 진짜 이렇게 왔다갔다 해야해..?

            else:
                continue

        os.chdir("../")

    return nouns_hash


def save_each_txt_count():
    # 각 파일에서 단어의 등장 횟수를 계산
    os.chdir("../data/txt/2.txt_files/")
    tagger = Mecab()
    nouns_hash = {}

    for subject in os.listdir(os.getcwd()):
        os.chdir("./" + subject)

        for subsection in  os.listdir(os.getcwd()):
            os.chdir("./" + subsection)

            for filename in os.listdir(os.getcwd()):
                nouns_hash.clear()
                if filename.endswith(".txt"):
                    # print(os.getcwd()+"\\"+filename)

                    f = open(filename, "r", encoding='utf-8')
                    data = f.read()
                    words_pos = tagger.pos(data)

                    for word in words_pos:
                        noun = word[0]
                        category = word[1]
                        if (category == "NNG"):
                            if noun in nouns_hash:
                                nouns_hash[noun] = nouns_hash[noun] + 1
                            else:
                                nouns_hash[noun] = 1
                    f.close()

                    # 여기에서 각 파일별로 txt파일 쓸 수 있게 수정. 경로 설정이 번거로울듯.
                    sorted_arr = sorted(nouns_hash.items(), key=operator.itemgetter(1), reverse=True)

                    os.chdir("../../../3.nouns_count/" + subject + "/" + subsection)

                    f = open(filename, "w", encoding='utf-8')
                    # for key, value in hash.items():
                    #     f.write(key +" "+ str(value)+"\n")
                    for item in sorted_arr:
                        f.write(item[0] + " " + str(item[1]) + "\n")
                    f.close()

                    os.chdir("../../../2.txt_files/" + subject + "/" + subsection)

                    # 경로 진짜 이렇게 왔다갔다 해야해..?

                else:
                    continue

            os.chdir("../")

        os.chdir("../")

    return nouns_hash


def save_one_txt_count():
    # 전체 파일에서 등장하는 단어의 등장 횟수를 계산
    os.chdir("../data/txt/3.nouns_count/")
    nouns_hash = {}

    for subject in os.listdir(os.getcwd()):
        os.chdir("./" + subject)

        for subsection in os.listdir(os.getcwd()):
            os.chdir("./" + subsection)

            for filename in os.listdir(os.getcwd()):
                if filename.endswith(".txt"):
                    # print(os.getcwd()+"\\"+filename)

                    fr = open(filename, "r", encoding='utf-8')
                    lines = fr.readlines()
                    for line in lines:
                        word = line.split(" ")
                        key = word[0]
                        value = int(word[1])

                        if key in nouns_hash:
                            nouns_hash[key] += value
                        else:
                            nouns_hash[key] = value
                    fr.close()

            os.chdir("../")

        os.chdir("../")

    sorted_arr = sorted(nouns_hash.items(), key=operator.itemgetter(1), reverse=True)

    fw = open("total_count.txt", "w", encoding='utf-8')

    for item in sorted_arr:
        fw.write(item[0] + " " + str(item[1]) + "\n")

    fw.close()

    return nouns_hash


def save_top_count(rank):
    # save_one_txt_count 함수에서 작성 한 파일을 이용, 등장 횟수 상위 rank개의 단어를 추출, 순위와 함께 저장.
    os.chdir("../data/txt/3.nouns_count/")

    fr = open("total_count.txt", "r", encoding='utf-8')
    fw = open("total_top_count.txt", "w", encoding='utf-8')

    for i in range(rank):
        line = fr.readline()
        word = line.split(" ")[0]

        fw.write(word + " " + str(i) + "\n")

    fr.close()
    fw.close()


def save_refined_rate(rank, test_set_rate, multiple_unit):
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
    os.chdir("../")

    print(total_rank_hash)

    os.chdir("./4.nouns_rate")
    for subject in os.listdir(os.getcwd()):
        os.chdir("./" + subject)

        for subsection in os.listdir(os.getcwd()):
            os.chdir("./" + subsection)

            for filename in os.listdir(os.getcwd()):
                if filename.endswith(".txt"):
                    if filename == "tmp.txt":
                        continue
                    refined_list = [0] * rank

                    f = open(filename, "r", encoding='utf-8')
                    lines = f.readlines()
                    for line in lines:
                        item = line.split(" ")
                        if item[0] in total_rank_hash:
                            # print(item)
                            refined_list[total_rank_hash[item[0]]] = item[1][:-1]

                    f.close()

                    # 여기에서 각 파일별로 txt파일 쓸 수 있게 수정. 경로 설정이 번거로울듯.
                    '''
                    if test_set_rate < random():
                        os.chdir("../../../5.refined_rate/train_set")
                    else:
                        os.chdir("../../../5.refined_rate/test_set")

                    f = open(filename, "w", encoding='utf-8')
                    
                    for item in refined_list:
                        f.write(str(item) + " ")
                    f.write("\n" + subject)
                    f.close()

                    os.chdir("../../4.nouns_rate/" + subject + "/" + subsection)
                    '''
                    # 경로 진짜 이렇게 왔다갔다 해야해..?
                    os.chdir("../../../5.refined_rate")

                    if test_set_rate < random():
                        f_data = open("train_set_data.txt", "a", encoding='utf-8')
                        f_label = open("train_set_label.txt", "a", encoding='utf-8')
                    else:
                        f_data = open("test_set_data.txt", "a", encoding='utf-8')
                        f_label = open("test_set_label.txt", "a", encoding='utf-8')

                    for item in refined_list:
                        f_data.write(str(float(item))[:8] + " ")
                    f_data.write("\n")

                    if subject == 'english':
                        f_label.write('1 0 0 0\n')
                    elif subject == 'korean':
                        f_label.write('0 1 0 0\n')
                    elif subject == 'mathematics':
                        f_label.write('0 0 1 0\n')
                    elif subject == 'science':
                        f_label.write('0 0 0 1\n')
                    else:
                        print("labeling error")

                    f_data.close()
                    f_label.close()

                    os.chdir("../4.nouns_rate/" + subject + "/" + subsection)

                else:
                    continue

            os.chdir("../")

        os.chdir("../")


if __name__ == "__main__":
    rank = 1000
    test_set_rate = 0.2
    multiple_unit = 100

    save_refined_rate(rank, test_set_rate, multiple_unit)

