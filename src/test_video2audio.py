import os

os.chdir("C:\\Users\\hello\\Desktop\\2016학년도수능 해설영상")

for filename in os.listdir(os.getcwd()):
    if filename.endswith(".mp4"): #or .avi, .mpeg, whatever.
        os.system("ffmpeg -y -i "+filename+" -vn -acodec libmp3lame -ar 16k -ac 1 -ab 64k "+filename[:-4]+".mp3")
    else:
        continue
