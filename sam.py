import cv2
import os
import numpy as np
import supervision as sv
from segment_anything import sam_model_registry, SamPredictor, SamAutomaticMaskGenerator
import more_itertools as mit
from shapely.geometry import Point, Polygon
import json
import math

with open("data.txt", "r") as fp:
    d = fp.readlines()
d = json.loads(d[0])

checkpoint = "/content/drive/MyDrive/sam_vit_h_4b8939.pth"
model_type = "vit_h"
sam = sam_model_registry[model_type](checkpoint=checkpoint)
sam.to(device='cuda')
predictor = SamPredictor(sam)
mask_generator = SamAutomaticMaskGenerator(sam)

directory = '/content/drive/MyDrive/test'
#P = [268.07,572,321.404,558,376.14,515,454.737,399,453.333,368,419.649,390,377.544,402,349.474,402,318.596,390,270.877,392,237.193,382,218.947,363,199.298,303,176.842,285,175.439,232,152.982,240,81.404,297,25.263,393,22.456,429,42.105,443,28.07,473,36.491,523,58.947,539,53.333,557,67.368,573,115.088,595,197.895,586,230.175,563,234.386,540,268.07,572]
#co = list(mit.grouper(P,2))
#print(co)
#poly = Polygon(co)
#PArr = np.zeros((640,640))
#for i in range(640):
#    for j in range(640):
#        p1 = Point(i,j)
#        if p1.within(poly):
#            PArr[i][j] = 1
#PArr = PArr[::-1]
metr = []
metr2 = []
for filename in os.listdir(directory):
    print(filename)
    if filename in d.keys():
      p1 = d[filename]
      aux = []
      aux2 = []
      for P in p1:
        co = list(mit.grouper(P,2))
        #print(co)
        poly = Polygon(co)
        PArr = np.zeros((640,640))
        for i in range(640):
            for j in range(640):
                p1 = Point(i,j)
                if p1.within(poly):
                    PArr[i][j] = 1
        f = os.path.join(directory,filename)
        image = cv2.imread(f)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        #sam_r = mask_generator.generate(image_rgb)
        #print(image_rgb.shape)
        box = np.array([0,0,0 + image.shape[1],0 + image.shape[0]])
        predictor.set_image(image_rgb)
        masks, scores, logits = predictor.predict(
        box=box,
        multimask_output=True
        )
        box_annotator = sv.BoxAnnotator(color=sv.Color.red())
        mask_annotator = sv.MaskAnnotator(color=sv.Color.red())

        detections = sv.Detections(
          xyxy=sv.mask_to_xyxy(masks=masks),
          mask=masks
        )
        detections = detections[detections.area == np.max(detections.area)]
        source_image = box_annotator.annotate(scene=image_rgb.copy(), detections=detections, skip_label=True)
        segmented_image = mask_annotator.annotate(scene=image_rgb.copy(), detections=detections)

        sv.plot_images_grid(
          images=[source_image, segmented_image],
          grid_size=(1, 2),
          titles=['source image', 'segmented image']
        )
        sv.plot_images_grid(
        images=masks,
        grid_size=(1, 4),
        size=(16, 4)
        )
        #aux = []
        for i in range(3):
          m = np.invert(masks[i])
          #print(m)
          dice = (np.sum(np.multiply(m,PArr))*2) / (np.sum(m) + np.sum(PArr))
          #print(dice)
          if len(aux) >= 3:
            aux[i] = aux[i] + dice
          else:
            aux.append(dice)
          count1 = np.count_nonzero(np.array(m + PArr)==1)
          count2 = np.count_nonzero(np.array(m + PArr)==2)
          IoU =  np.sum(np.multiply(m,PArr)) / (count1 + count2)
          if len(aux2) >= 3:
            aux2[i] = aux2[i] + IoU
          else:
            aux2.append(IoU)
      am = np.amax(aux)
      am2 = np.amax(aux2)
      metr.append(am)
      metr2.append(am2)
      if am < 0.5:
        print(filename)
print("Promedio DICE")
print(metr)
print(np.mean(metr))
print("Promedio IoU")
print(metr2)
print(np.mean(metr2))