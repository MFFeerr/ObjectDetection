import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage as ndi
from PIL import Image
#

from skimage import color, filters
from skimage.filters import rank
from skimage.morphology import disk
#
from skimage.segmentation import watershed
from skimage.feature import peak_local_max
from skimage.util import img_as_ubyte

#
# Generate an initial image with two overlapping circles
directory = 'C:/Users/HP/PycharmProjects/ProyectoFinal/Crop20'
num = 0
for filename in os.listdir(directory):
    f = os.path.join(directory,filename)
    img = Image.open(f)
    gray = color.rgb2gray(img)
    print(filename)
    # Apply thresholding to convert the image to binary
    thresh = filters.threshold_otsu(gray)
    binary = gray <= thresh

    distance = ndi.distance_transform_edt(binary)
    coords = peak_local_max(distance, footprint=np.ones((3, 3)), labels=binary)
    mask = np.zeros(distance.shape, dtype=bool)
    mask[tuple(coords.T)] = True
    markers, _ = ndi.label(mask)
    labels = watershed(-distance, markers, mask=binary)

    fig, axes = plt.subplots(ncols=3, figsize=(9, 3), sharex=True, sharey=True)
    ax = axes.ravel()

    ax[0].imshow(binary, cmap=plt.cm.gray)
    ax[0].set_title('Overlapping objects')
    ax[1].imshow(-distance, cmap=plt.cm.gray)
    ax[1].set_title('Distances')
    ax[2].imshow(labels, cmap=plt.cm.nipy_spectral)
    ax[2].set_title('Separated objects')

    for a in ax:
        a.set_axis_off()

    fig.tight_layout()
    plt.show()