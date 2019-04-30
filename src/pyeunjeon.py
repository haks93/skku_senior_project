from eunjeon import Mecab
import os
import operator

def pyeunjeon(subjectIndex):
    # 0: kor, 1: eng,  2: math, 3: physics, 4:chemical, 5: bio, 6: earth
    subj = ["korean", "english", "mathematics", "sci_physics", "sci_chemical", "sci_bioscience", "sci_earthscience"]

    subject = subj[subjectIndex]

    os.chdir("../data/txt/2.txt_files/"+subject)
    tagger = Mecab()
    hash = {}

    for filename2 in  os.listdir(os.getcwd()):
        os.chdir("./" + filename2)

        for filename in os.listdir(os.getcwd()):
            if filename.endswith(".txt"):
                print(os.getcwd()+"\\"+filename)

                f = open(filename, "r", encoding='utf-8')
                data = f.read()
                words_pos = tagger.pos(data)

                for word in words_pos:
                    noun = word[0]
                    category = word[1]
                    if (category == 'NNG'):
                        if noun in hash:
                            hash[noun] = hash[noun] + 1
                        else:
                            hash[noun] = 1
                f.close()
            else:
                continue

        os.chdir("../")

    sortedArr = sorted(hash.items(), key=operator.itemgetter(1), reverse=True)

    os.chdir("../../3.nouns/")
    f = open(subject+'.txt', "w", encoding='utf-8')
    # for key, value in hash.items():
    #     f.write(key +" "+ str(value)+"\n")

    for item in sortedArr:
        f.write(item[0] + " " + str(item[1]) + "\n")

    f.close()


    return hash

if __name__ == "__main__":

    pyeunjeon(7)
