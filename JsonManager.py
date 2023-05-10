import json

f =open('C:/Users/HP/PycharmProjects/ProyectoFinal/_annotations.coco.json')

data = json.load(f)

info = data["images"]
ims = {}
for i in info:
    ims[i["id"]] = i["file_name"]

annot = data["annotations"]
final = {}
for i in annot:
    print(i["image_id"])
    n = ims[i["image_id"]]
    if n not in final.keys():
        final[n] = []
    final[n].append(i["segmentation"][0])

with open("data2.txt", "w") as fp:
    json.dump(final,fp)
#    d = fp.readlines()

#d = json.loads(d[0])

#for i in d.keys():
#    print(i)
