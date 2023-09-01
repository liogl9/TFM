import cv2
import matplotlib.pyplot as plt
import os
import numpy as np
import copy
import random
from PIL import Image

image = cv2.imread('hist_og.png')
uimage = copy.deepcopy(image)

# display the image
fig, ax = plt.subplots(3, 2)
# plt.imshow(image)
ax[0, 0].imshow(image)
colors = ('red', 'green', 'blue')

pos = list(range(600))
random.shuffle(pos)

for _ in range(600):
    uimage = image[pos, :, :]
random.shuffle(pos)
for _ in range(600):
    uimage = uimage[:, pos, :]

ax[2, 0].imshow(uimage)
pil_uimage = Image.fromarray(np.uint8(uimage)).convert('RGB')

hist = pil_uimage.histogram()

hr = hist[0:256]
hb = hist[256:256*2]
hg = hist[256*2:256*3]

for i in range(256):
    ax[0, 1].bar(i, hr[i], color='red')
    ax[1, 1].bar(i, hg[i], color='green')
    ax[2, 1].bar(i, hb[i], color='blue')

ax[0, 1].set_title("Red Histogram")
ax[1, 1].set_title("Green Histogram")
ax[2, 1].set_title("Blue Histogram")
ax[0, 0].set_title("Original image")
ax[0, 0].axis("off")
ax[2, 0].set_title("Shuffled image")
ax[2, 0].axis("off")
# fig.delaxes(ax[1, 0])
ax[1, 0].set_visible(False)
plt.show()
