import os
import statistics
import cv2
from ultralytics import YOLO
import numpy as np
from PIL import Image

def box_l(image,box,color=(128,128,128)):
        lw = max(round(sum(image.shape) / 2 * 0.003), 2)
        p1, p2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
        with open('ImagesPoints20.txt', 'a') as f:
            f.write(str(box[0])+','+str(box[1])+','+str(box[2])+','+str(box[3])+'\n')
        cv2.rectangle(image, p1, p2, color, thickness=lw, lineType=cv2.LINE_AA)
        return int(box[0]), int(box[1]), int(box[2]), int(box[3])

def BBmetrics(a,b,c,d,m,n,o,p):
    xA = max(a, m)
    yA = max(b, n)
    xB = min(c, o)
    yB = min(d, p)
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    boxAArea = (c - a + 1) * (d - b + 1)
    boxBArea = (o - m + 1) * (p - n + 1)
    iou = interArea / float(boxAArea + boxBArea - interArea)
    dice = (2*interArea) / float(boxAArea+boxBArea)
    return iou,dice

model = YOLO("yolov8n.pt")
model.train(data="C:/Users/HP/PycharmProjects/TesisT/Dataset/data.yaml", epochs=2)
metrics = model.val()  # evaluate model performance on the validation set
directory = 'C:/Users/HP/PycharmProjects/TesisT/Dataset/test/images'
dic2 = 'C:/Users/HP/PycharmProjects/TesisT/Dataset/test/labels'
num = 0
IOULi = []
DICELi = []
for filename in os.listdir(directory):
    f = os.path.join(directory,filename)
    im = Image.open(f)
    im = np.asarray(im)
    res = model(im)
    res = list(res)[0]
    # YOLOplot(im, res.boxes.boxes,num)
    aux1 = []
    aux2 = []
    boxn = len(res.boxes.boxes)
    print("LEN:  " + str(boxn))
    for box in res.boxes.boxes:
        print(box[4])
        x1,y1,x2,y2 = box_l(im, box)
        for filename2 in os.listdir(dic2):
            if filename[:-4] in filename2:
                print("found")
                with open(os.path.join(dic2, filename2), 'r') as f:
                    a = f.readlines()
                    for line in a:
                        a1 = line.split()
                        w, h = im.shape[1], im.shape[0]
                        x = float(a1[1]) * w
                        y = float(a1[2]) * h
                        w1 = float(a1[3]) * w
                        h1 = float(a1[4]) * h
                        #p1, p2 = (int(x + w1 / 2), int(y - h1 / 2)), (int(x - w1 / 2), int(y + h1 / 2))
                        g, k = BBmetrics(x1, y1, x2, y2, int(x - w1 / 2), int(y - h1 / 2), int(x + w1 / 2),int(y + h1 / 2))
                        aux1.append(g)
                        aux2.append(k)
                        print(str(x1) + " " + str(y1) + " " + str(x2) + " " + str(y2))
                        print(str(x - w1 / 2) + " " + str(y - h1 / 2)+ " " +str(x + w1 / 2)+ " " + str(y + h1 / 2))
                        print("IOU ----------------------------------------")
                        print(g)
                        print(k)
    aux2.sort()
    aux1.sort()
    IOULi = IOULi + aux1[:boxn]
    DICELi = DICELi + aux2[:boxn]
                    #cv2.rectangle(im, p1, p2, (128, 128, 128), lineType=cv2.LINE_AA)
                    #image = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
                    #cv2.imshow('img', image)
                    #cv2.waitKey(0)

    image = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    cv2.imwrite('ResultsY20/result' + str(num) + '.jpg', image)
    cv2.waitKey(0)
    num+=1
    if num == 30:
        break
print(statistics.mean(IOULi))
print(statistics.mean(DICELi))
