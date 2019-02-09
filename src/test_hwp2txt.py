# -*- coding: utf-8 -*-

'''
Run with python 2.7
'''

import os

print(os.getcwd())

os.chdir("../data/txt")

print(os.getcwd())

f = open("test.hwp", "r")

text = f.read()
print(text)
f.close()

#os.chdir("../data/txt/ebsi_고3/고3_국어/test_file")

'''



for filename in os.listdir(os.getcwd()):
    if filename.endswith(".hwp"):
        os.system("hwp5txt "+filename)
    else:
        continue
'''