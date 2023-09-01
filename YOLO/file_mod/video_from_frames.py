
import cv2
import os
from tqdm import tqdm

crops = []

X_MIN = 2500
Y_MIN = 600

# Load your crops
imgs_path = os.path.join(os.getcwd(), "frames_png")
if os.path.isdir(imgs_path):
    for file in sorted(os.listdir(imgs_path)):
        if file.endswith(".png"):
            img_path = os.path.join(imgs_path, str(file))
            img = cv2.imread(img_path)
            crops.append(img)

rows, cols = img.shape[:2]

added_imgs = []

# Add your crops (Lio, debes cambiar X_MIN e Y_MIN para cambiar las posiciones)
for crop in tqdm(crops):
    base_path = os.path.join(os.getcwd(), "fondo_3.png")
    base_img = cv2.imread(base_path)
    roi = base_img[Y_MIN:rows+Y_MIN, X_MIN:cols+X_MIN]

    gray_img = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray_img, 10, 255, cv2.THRESH_BINARY)
    mask_inv = cv2.bitwise_not(mask)

    base_img_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)
    img_fg = cv2.bitwise_and(crop, crop, mask=mask)

    dst = cv2.add(base_img_bg, img_fg)
    base_img[Y_MIN:rows+Y_MIN, X_MIN:cols+X_MIN] = dst

    added_imgs.append(base_img)

height, width = base_img.shape[:2]
