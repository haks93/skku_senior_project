from eunjeon import Mecab
import os

subject = "mathematics" #just change this.

os.chdir("../data/txt/2.txt_files/"+subject)
tagger = Mecab()
hash = {}

for filename in os.listdir(os.getcwd()):
    if filename.endswith(".txt"):
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

os.chdir("../../3.nouns/")
f = open(subject+'.txt', "a")
for key, value in hash.items():
    f.write(key +" "+ str(value)+"\n")

f.close()

print(hash)

