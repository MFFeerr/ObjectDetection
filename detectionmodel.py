import os
import statistics
import cv2
from ultralytics import YOLO
import numpy as np
from PIL import Image

def box_l(image,box,n,ep,color=(128,128,128)):
        lw = max(round(sum(image.shape) / 2 * 0.003), 2)
        p1, p2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
        #with open('ImagesPointsA.txt', 'a') as f:
        #    f.write(str(n)+ ","+str(box[0])+','+str(box[1])+','+str(box[2])+','+str(box[3])+'\n')
        crop_i = image[int(box[1]):int(box[3]), int(box[0]):int(box[2])]
        crop_i = cv2.cvtColor(crop_i, cv2.COLOR_BGR2RGB)
        file_name = "ImagesF" + str(ep) + "/CropIm" + str(n) +"_"+ str(box[0])+ '.jpg'
        cv2.imwrite(file_name, crop_i)
        cv2.waitKey(0)
        cv2.rectangle(image, p1, p2, color, thickness=lw, lineType=cv2.LINE_AA)
        return int(box[0]), int(box[1]), int(box[2]), int(box[3]), file_name

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

def saveInFinal(a):
    with open('FinalMetric.txt', 'a') as f:
        f.write(a)
ep = [1]
for yan in range(1,11):
    FilIname = [[],[],[],[],[],[],[],[],[],[],[]]
    for e in ep:
        print("----" + str(e) + "----")
        model = YOLO("yolov8n.pt")
        model.train(data="C:/Users/HP/PycharmProjects/ProyectoFinal/DatasetF/data" + str(yan) +".yaml", epochs=int(e))
        metrics = model.val()  # evaluate model performance on the validation set
        directory = 'C:/Users/HP/PycharmProjects/ProyectoFinal/DatasetF/test' + str(yan) +'/images'
        dic2 = 'C:/Users/HP/PycharmProjects/ProyectoFinal/DatasetF/test' + str(yan) +'/labels'
        num = 0
        noFound = 0
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
            if boxn == 0:
                noFound= noFound + 1
            for box in res.boxes.boxes:
                if float(box[-2]) >= 0.5:
                    print("enter---" + str(box[-2]))
                    x1,y1,x2,y2,nom = box_l(im, box,num,yan)
                    if float(box[-2]) >= 0.5:
                        FilIname[0].append(nom)
                    if float(box[-2]) >= 0.55:
                        FilIname[1].append(nom)
                    if float(box[-2]) >= 0.6:
                        FilIname[2].append(nom)
                    if float(box[-2]) >= 0.65:
                        FilIname[3].append(nom)
                    if float(box[-2]) >= 0.7:
                        FilIname[4].append(nom)
                    if float(box[-2]) >= 0.75:
                        FilIname[5].append(nom)
                    if float(box[-2]) >= 0.8:
                        FilIname[6].append(nom)
                    if float(box[-2]) >= 0.85:
                        FilIname[7].append(nom)
                    if float(box[-2]) >= 0.9:
                        FilIname[8].append(nom)
                    if float(box[-2]) >= 0.95:
                        FilIname[9].append(nom)
                    if float(box[-2]) >= 1:
                        FilIname[10].append(nom)
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
                                    print(filename2)
                                    print(str(x1) + " " + str(y1) + " " + str(x2) + " " + str(y2))
                                    print(str(x - w1 / 2) + " " + str(y - h1 / 2)+ " " +str(x + w1 / 2)+ " " + str(y + h1 / 2))
                                    print("IOU ----------------------------------------")
                                    print(g)
                                    print(k)
            aux2.sort(reverse=True)
            print(aux2)
            print(aux2[:boxn])
            aux1.sort(reverse=True)
            print(aux1)
            print(aux1[:boxn])
            IOULi = IOULi + aux1[:boxn]
            DICELi = DICELi + aux2[:boxn]
                        #cv2.rectangle(im, p1, p2, (128, 128, 128), lineType=cv2.LINE_AA)
                        #image = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
                        #cv2.imshow('img', image)
                        #cv2.waitKey(0)

            image = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
            #cv2.imwrite("ResultsT"+str(e)+"/result" + str(num) + '.jpg', image)
            #cv2.waitKey(0)
            num+=1
        print(IOULi)
        print(DICELi)
        print(statistics.mean(IOULi))
        print(statistics.mean(DICELi))
        print(noFound)
        #success = model.export(format="onnx")
        saveInFinal(str(yan) + ": IoU: " + str(statistics.mean(IOULi)) + ",DICE: " + str(statistics.mean(DICELi)) + ',NF: ' + str(noFound) + '\n')
    for i in FilIname:
        with open('ImagesFileName' + str(yan) + '.txt', 'a') as f:
            f.write(str(i) + '\n')
