import os
import numpy as np
from skimage.color import rgb2gray
from skimage.filters import gaussian
from skimage.segmentation import active_contour
import more_itertools as mit
from shapely.geometry import Point, Polygon
import json
from PIL import Image
import matplotlib.pyplot as plt
with open("C:/Users/HP/PycharmProjects/ProyectoFinal/data2.txt", "r") as fp:
    d = fp.readlines()
d = json.loads(d[0])


directory = 'C:/Users/HP/PycharmProjects/ProyectoFinal/test'
metr = []
metr2 = []
for filename in os.listdir(directory):
    print(filename)
    if filename in d.keys():
      f = os.path.join(directory, filename)
      im = Image.open(f)
      imt = rgb2gray(im)
      w, h = im.size
      p1 = d[filename]
      aux = []
      aux2 = []
      for P in p1:
        co = list(mit.grouper(P,2))
        #print(co)
        poly = Polygon(co)
        PArr = np.zeros((w,h))
        for i in range(w):
            for j in range(h):
                p1 = Point(i,j)
                if p1.within(poly):
                    PArr[i][j] = 1
        print(w)
        print(h)

        num = 300
        n1 = (num / 2) * w / (w + h)
        n2 = (num / 2) * h / (w + h)
        #print(n1)
        #print(n2)
        v = np.concatenate(
          (np.linspace(0, w, int(n1)), np.linspace(w, w, int(n2)), np.linspace(w, 0, int(n1)), np.zeros(int(n2))))
        ho = np.concatenate(
          (np.zeros(int(n1)), np.linspace(0, h, int(n2)), np.linspace(h, h, int(n1)), np.linspace(h, 0, int(n2))))
        init = np.array([ho, v]).T
        snake = active_contour(gaussian(imt, 2, preserve_range=False),
                               init, alpha=0.040, beta=10, gamma=0.001)

        aux11 = []
        for i in snake:
          aux11.append(i[1])
          aux11.append(i[0])
        co2 = list(mit.grouper(aux11, 2))
        # print(co)
        poly2 = Polygon(co2)
        PArr2 = np.zeros((w, h))
        for i in range(w):
          for j in range(h):
            p2 = Point(i, j)
            if p2.within(poly2):
              PArr2[i][j] = 1
        #print(m)
        dice = (np.sum(np.multiply(PArr2,PArr))*2) / (np.sum(PArr2) + np.sum(PArr))
        #print(dice)
        if len(aux) >= 1:
          aux[0] = aux[0] + dice
        else:
          aux.append(dice)
        count1 = np.count_nonzero(np.array(PArr2 + PArr)==1)
        count2 = np.count_nonzero(np.array(PArr2 + PArr)==2)
        IoU =  np.sum(np.multiply(PArr2,PArr)) / (count1 + count2)
        if len(aux2) >= 1:
          aux2[0] = aux2[0] + IoU
        else:
          aux2.append(IoU)
      print(aux)
      print(aux2)
      am = np.amax(aux)
      am2 = np.amax(aux2)
      metr.append(am)
      metr2.append(am2)
      #if am < 0.5:
      #  print(filename)
print("Promedio DICE")
print(metr)
print(np.mean(metr))
print("Promedio IoU")
print(metr2)
print(np.mean(metr2))