import os
import cv2
from ultralytics import YOLO
import numpy as np
from PIL import Image

def box_l(image,box,color=(128,128,128)):
        lw = max(round(sum(image.shape) / 2 * 0.003), 2)
        p1, p2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
        with open('ImagesPoints3.txt', 'a') as f:
            f.write(str(box[0])+','+str(box[1])+','+str(box[2])+','+str(box[3])+'\n')
        cv2.rectangle(image, p1, p2, color, thickness=lw, lineType=cv2.LINE_AA)

def YOLOplot(image, boxes,n):
        for box in boxes:
            box_l(image,box)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        #cv2.imshow('img',image)
        cv2.imwrite('ResultsY3/result'+str(n)+'.jpg',image)
        cv2.waitKey(0)

model = YOLO("yolov8n.pt")
model.train(data="C:/Users/HP/PycharmProjects/TesisT/Dataset/data.yaml", epochs=3)
metrics = model.val()  # evaluate model performance on the validation set
directory = 'C:/Users/HP/PycharmProjects/TesisT/Dataset/train/images'
num = 0
for filename in os.listdir(directory):
    f = os.path.join(directory,filename)
    im = Image.open(f)
    im = np.asarray(im)
    res = model(im)
    res = list(res)[0]
    YOLOplot(im, res.boxes.boxes,num)
    num+=1
