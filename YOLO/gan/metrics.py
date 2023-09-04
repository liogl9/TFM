import warnings
# from skimage.measure import compare_ssim
# from skimage.transform import resize
from scipy.stats import wasserstein_distance
# from scipy.misc import imsave
import numpy as np
import cv2
import matplotlib.pyplot as plt

##
# Globals
##

warnings.filterwarnings('ignore')

# specify resized image sizes
height = 2**10
width = 2**10

##
# Functions
##


def get_img(path, norm_size=True, norm_exposure=False):
    '''
    Prepare an image for image processing tasks
    '''
    # flatten returns a 2d grayscale array
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    # resizing returns float vals 0:255; convert to ints for downstream tasks
    # if norm_size:
    #     img = resize(img, (height, width),
    #                  anti_aliasing=True, preserve_range=True)
    if norm_exposure:
        img = normalize_exposure(img)
    return img


def get_histogram(img):
    '''
    Get the histogram of an image. For an 8-bit, grayscale image, the
    histogram will be a 256 unit vector in which the nth value indicates
    the percent of the pixels in the image with the given darkness level.
    The histogram's values sum to 1.
    '''
    h, w = img.shape
    hist = [0.0] * 256
    for i in range(h):
        for j in range(w):
            hist[img[i, j]] += 1
    return np.array(hist) / (h * w)


def normalize_exposure(img):
    '''
    Normalize the exposure of an image.
    '''
    img = img.astype(int)
    hist = get_histogram(img)
    # get the sum of vals accumulated by each position in hist
    cdf = np.array([sum(hist[:i+1]) for i in range(len(hist))])
    # determine the normalization values for each unit of the cdf
    sk = np.uint8(255 * cdf)
    # normalize each position in the output image
    height, width = img.shape
    normalized = np.zeros_like(img)
    for i in range(0, height):
        for j in range(0, width):
            normalized[i, j] = sk[img[i, j]]
    return normalized.astype(int)


def earth_movers_distance(path_a, path_b):
    '''
    Measure the Earth Mover's distance between two images
    @args:
      {str} path_a: the path to an image file
      {str} path_b: the path to an image file
    @returns:
      TODO
    '''
    img_a = get_img(path_a, norm_exposure=True)
    img_b = get_img(path_b, norm_exposure=True)
    hist_a = get_histogram(img_a)
    hist_b = get_histogram(img_b)
    return wasserstein_distance(hist_a, hist_b)


if __name__ == '__main__':
    img_a = './datasets/G1_a_real_multi_0_5/images/train/000001.png'
    # Dataset distinto
    # img_b = './datasets/G1_a_sin/images/train/000006.png'
    # Mismo dataset
    img_b = './datasets/G1_a_real_multi_0_5/images/train/000042.png'
    # get the similarity values
    # structural_sim = structural_sim(img_a, img_b)
    # pixel_sim = pixel_sim(img_a, img_b)
    # sift_sim = sift_sim(img_a, img_b)
    emd = earth_movers_distance(img_a, img_b)
    print(emd)
