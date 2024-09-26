import os
import numpy as np
import cv2

db_src = "/media/hd0/wuguoming/dataset/ucf101/"
label = os.listdir(db_src + "videos/")
'''
print(label)
labelfile = open("F:\\workspace\\dataset\\hmdb51_label.txt", "w")
for s in label:
    labelfile.write(s+"\n")
labelfile.close()
'''
labeltot = len(label)
labelnow = 1
for label_name in label:
    video_src = db_src + "videos/" + label_name + "/"
    videos = os.listdir(video_src)
    tot = len(videos)

    now = 0
    for v in videos:
        cap = cv2.VideoCapture(video_src + v)
        v = v.split('.')[0]
        cnt = 1
        'img_{:05d}.jpg'
        flag = True
        while flag:
            flag, frame = cap.read()
            if flag:
                tar = db_src + "rawframes/" + label_name + "/" + v + "/"
                if not os.path.exists(tar):
                    os.makedirs(tar)
                cv2.imwrite(tar + "img_%05d.jpg" % cnt, frame)
                cnt += 1
        cap.release()

        if now % 5 == 0:
            print('%.2f%% %s %.2f%%' % (labelnow*100/labeltot, label_name, now*100/tot))
        now += 1

    labelnow += 1