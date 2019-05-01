from eunjeon import Mecab
import os
import operator

def pyeunjeon(subjectIndex):
    # 0: kor, 1: eng,  2: math, 3: science
    subj = ["korean", "english", "mathematics", "science"]

    subject = subj[subjectIndex]

    os.chdir("../data/txt/2.txt_files/"+subject)
    tagger = Mecab()
    nouns_hash = {}

    for subsection in  os.listdir(os.getcwd()):
        os.chdir("./" + subsection)

        for filename in os.listdir(os.getcwd()):
            if filename.endswith(".txt"):
                print(os.getcwd()+"\\"+filename)

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

                # 여기에서 각 파일별로 txt파일 출력할 수 있게 수정. 경로 설정이 번거로울듯.
                os.chdir("../../../3.nouns/" + subject + "/" + subsection)
                # 진짜 이렇게 왔다갔다 해야해..?

            else:
                continue

        os.chdir("../")

    sortedArr = sorted(nouns_hash.items(), key=operator.itemgetter(1), reverse=True)

    os.chdir("../../3.nouns/")
    f = open(subject+'.txt', "w", encoding='utf-8')
    # for key, value in hash.items():
    #     f.write(key +" "+ str(value)+"\n")

    for item in sortedArr:
        f.write(item[0] + " " + str(item[1]) + "\n")

    f.close()


    return nouns_hash

if __name__ == "__main__":

    pyeunjeon(7)
