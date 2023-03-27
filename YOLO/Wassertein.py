from scipy.stats import wasserstein_distance
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import copy
import random
from PIL import Image

# image to signature for color image


def img_to_sig(img):
    sig = np.empty((img.size, 4), dtype=np.float32)
    idx = 0
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            for k in range(img.shape[2]):
                sig[idx] = np.array([img[i, j, k], i, j, k])
                idx += 1
    return sig

# image = cv2.imread('hist_og.png')
# uimage = copy.deepcopy(image)


# # display the image
# fig, ax = plt.subplots(3, 2)
# # plt.imshow(image)
# # ax[0, 0].imshow(image)
# colors = ('red', 'green', 'blue')

# pos = list(range(600))
# random.shuffle(pos)

# # for _ in range(600):
# uimage = image[pos, :, :]
# random.shuffle(pos)
# # for _ in range(600):
# uimage = uimage[:, pos, :]

# cv2.imwrite('shufled_img.png', uimage)
# # ax[2, 0].imshow(uimage)

# pil_uimage = Image.fromarray(np.uint8(uimage)).convert('RGB')
# pil_image = Image.fromarray(np.uint8(image)).convert('RGB')

# hist_uimage = pil_uimage.histogram()
# hist_image = pil_image.histogram()


# hr_uimage = hist_uimage[0:256]
# hb_uimage = hist_uimage[256:256*2]
# hg_uimage = hist_uimage[256*2:256*3]

# hr_image = hist_image[0:256]
# hb_image = hist_image[256:256*2]
# hg_image = hist_image[256*2:256*3]

# for i in range(256):
#     ax[0, 1].bar(i, hr_uimage[i], color='red')
#     ax[1, 1].bar(i, hg_uimage[i], color='green')
#     ax[2, 1].bar(i, hb_uimage[i], color='blue')

#     ax[0, 0].bar(i, hr_image[i], color='red')
#     ax[1, 0].bar(i, hg_image[i], color='green')
#     ax[2, 0].bar(i, hb_image[i], color='blue')


# ax[0, 1].set_title("Red Histogram")
# ax[1, 1].set_title("Green Histogram")
# ax[2, 1].set_title("Blue Histogram")
# ax[0, 0].set_title("Original image")
# ax[0, 0].axis("off")
# ax[2, 0].set_title("Shuffled image")
# ax[2, 0].axis("off")
# # fig.delaxes(ax[1, 0])
# # ax[1, 0].set_visible(False)
# plt.show()
img1 = cv2.imread('./datasets/G1_a_real_multi_0_5/images/train/000001.png')
img2 = cv2.imread('./datasets/G1_a_sin/images/train/000006.png')
img1 = cv2.resize(img1, [30, 30])
img2 = cv2.resize(img2, [30, 30])
sig1 = img_to_sig(img1)
sig2 = img_to_sig(img2)
distance, lowerbound, flow_matrix = cv2.EMD(
    sig1, sig2, cv2.DIST_L2, lowerBound=0)
print(distance, lowerbound, flow_matrix)
