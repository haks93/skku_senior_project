import os

os.chdir("C:\\Users\\hello\\Desktop\\2016학년도수능 해설영상")

os.system("ffmpeg -y -i 물리I(1)_(고3-공통).mp3 -ss 0 -t 59 -acodec copy test.mp3")
