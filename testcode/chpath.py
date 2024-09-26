import os
'''
src = "/media/hd0/wuguoming/dataset/ucf101/"
filename = "ucf101_val_split_1_videos.txt"

tar = "buaojie/data/ucf101/videos"
res = "wuguoming/dataset/ucf101/rawframes"

file = open(src + filename, encoding="utf-8")
lines = file.readlines()
file.close()

file = open(src + filename, "w")
for line in lines:
    print(line)
    line = line.replace(tar, res)
    line = line.replace(".avi", "")
    print(line)

    file.write(line)

file.close()
'''

file = open('/media/hd0/wuguoming/dataset/ucf101/ucf101_val_split_1_videos.txt', encoding="utf-8", )
lines = file.readlines()
file.close()

file = open('/media/hd0/wuguoming/dataset/ucf101/ucf101_val_split_1_videos.txt', "w")
newfile = []
for line in lines:
    line = line.strip().split(' ')
    path = line[0]
    label = line[1]

    tar = os.listdir(path)
    num = len(tar)
    newline = path + " " + str(num) + " " + label + "\n"
    print(newline)
    file.write(newline)
file.close()
