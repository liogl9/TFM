from copy import deepcopy
from random import randint
import cv2
import os
from cv2 import line
from tqdm import tqdm
import sys
import numpy as np
from array import array

X_MIN = 0
Y_MIN = 0
show = False
save = True


def draw_bboxes(image: array, bboxes, categories) -> array:
    """
    Draw bboxes on a image given the category and the bboxes

    Args:
        - image: to be used as template to draw on
        - bboxes: list of [(x, y, width, height)]
        - categories: full list of coco categories

    Returns:
        - bboxes_image: copy of image with bboxes drawn on top
    """
    bboxes_image = deepcopy(image)
    for i in range(len(bboxes)):
        category = categories[0]
        # Change of format: x1,y1,h,w -> x1,y1,x2,y2
        bbox = [bboxes[i][0], bboxes[i][1],
                (bboxes[i][0] + bboxes[i][2]), (bboxes[i][1] + bboxes[i][3])]
        center = [int((bbox[0] + bbox[2]) / 2), int((bbox[1] + bbox[3]) / 2)]
        cv2.rectangle(
            bboxes_image, (bbox[0], bbox[1]),
            (bbox[2], bbox[3]), (0, 0, 255), 2)
        cv2.rectangle(
            bboxes_image, (bbox[0], bbox[1]-30),
            (bbox[0]+80, bbox[1]), (0, 0, 255), -1)
        cv2.putText(bboxes_image, category,
                    (bbox[0], bbox[1]-5), 0, 1, (255, 255, 255), 2)
    return bboxes_image


def move_up(img_fg, mask, lbl_og, mov_px):
    img_mov = deepcopy(img_fg)
    # Mask for moving the crop
    mask_mov = deepcopy(mask)

    for i in range(img_fg.shape[0]):
        if i < img_fg.shape[0]-mov_px:
            img_mov[i, :, :] = img_mov[i+mov_px, :, :]
            mask_mov[i, :] = mask[i+mov_px, :]
        else:
            img_mov[i, :, :] = 0
            mask_mov[i, :] = 255

    lbl_mov = deepcopy(lbl_og)
    lbl_mov[2] -= mov_px*1.0/img_fg.shape[0]

    return img_mov, mask_mov, lbl_mov


def move_down(img_fg, mask, lbl_og, mov_px):
    img_mov = deepcopy(img_fg)
    # Mask for moving the crop
    mask_mov = deepcopy(mask)
    for i in range(img_fg.shape[0]-1, -1, -1):
        if i > mov_px:
            img_mov[i, :, :] = img_mov[i-mov_px, :, :]
            mask_mov[i, :] = mask[i-mov_px, :]
        else:
            img_mov[i, :, :] = 0
            mask_mov[i, :] = 255

    lbl_mov = deepcopy(lbl_og)
    lbl_mov[2] += mov_px*1.0/img_fg.shape[0]

    return img_mov, mask_mov, lbl_mov


def move_left(img_fg, mask, lbl_og, mov_px):
    img_mov = deepcopy(img_fg)
    # Mask for moving the crop
    mask_mov = deepcopy(mask)
    for i in range(img_fg.shape[1]):
        if i < img_fg.shape[1]-mov_px:
            img_mov[:, i, :] = img_mov[:, i+mov_px, :]
            mask_mov[:, i] = mask[:, i+mov_px]
        else:
            img_mov[:, i, :] = 0
            mask_mov[:, i] = 255

    lbl_mov = deepcopy(lbl_og)
    lbl_mov[1] -= mov_px*1.0/img_fg.shape[1]

    return img_mov, mask_mov, lbl_mov


def move_right(img_fg, mask, lbl_og, mov_px):
    img_mov = deepcopy(img_fg)
    # Mask for moving the crop
    mask_mov = deepcopy(mask)
    for i in range(img_fg.shape[1]-1, -1, -1):
        if i > mov_px:
            img_mov[:, i, :] = img_mov[:, i-mov_px, :]
            mask_mov[:, i] = mask[:, i-mov_px]
        else:
            img_mov[:, i, :] = 0
            mask_mov[:, i] = 255

    lbl_mov = deepcopy(lbl_og)
    lbl_mov[1] += mov_px*1.0/img_fg.shape[1]

    return img_mov, mask_mov, lbl_mov


if __name__ == "__main__":
    crops = []
    crops_path = []
    dataset_dir = os.path.join(os.getcwd(), "datasets", "G1_a_real")
    dataset_dst_dir = os.path.join(
        os.getcwd(), "datasets", "G1_a_real_multi_0_5")

    lbl_og_dir = os.path.join(dataset_dir, "labels")
    lbl_dst_dir = os.path.join(dataset_dst_dir, "labels")
    # Load your crops
    imgs_path = os.path.join(dataset_dir, "images")

    imgs_dst_dir = os.path.join(dataset_dst_dir, "images")
    bbox_dst_dir = os.path.join(dataset_dst_dir, "bboxes")

    if not os.path.exists(lbl_dst_dir):
        os.makedirs(lbl_dst_dir)
        os.makedirs(os.path.join(lbl_dst_dir, 'train'))
        os.makedirs(os.path.join(lbl_dst_dir, 'test'))
    if not os.path.exists(imgs_dst_dir):
        os.makedirs(imgs_dst_dir)
        os.makedirs(os.path.join(imgs_dst_dir, 'train'))
        os.makedirs(os.path.join(imgs_dst_dir, 'test'))
    if not os.path.exists(bbox_dst_dir):
        os.makedirs(bbox_dst_dir)

    padded_path = os.path.join(
        dataset_dir, "imgs_padd.txt")

    with open(padded_path) as f:
        padded_imgs_list = [line.strip() for line in f.readlines()]
    padded_imgs_list.remove('Name')
    try:
        padded_imgs_list.remove('')
    except:
        pass

    if os.path.isdir(imgs_path):
        for file in sorted(os.listdir(imgs_path)):
            if file.endswith(".png") and file not in padded_imgs_list:
                img_path = os.path.join(imgs_path, str(file))
                img = cv2.imread(img_path)
                crops.append(img)
                crops_path.append(img_path)

    rows, cols = img.shape[:2]

    added_imgs = []
    num_crops = 1000

    for i in tqdm(range(num_crops)):
        # Path for the base img
        base_path = os.path.join(os.getcwd(), "fotos_base", "base_new.png")
        # Load the base img
        base_img = cv2.imread(base_path)
        # Specify the area of interest to paste the crops
        roi = base_img[Y_MIN:rows+Y_MIN, X_MIN:cols+X_MIN]
        # Random number of crops to add (3,4 -> 0.6, 0.4)
        num_pieces = np.random.choice(
            np.arange(0, 5), 1, p=[0.05, 0.10, 0.20, 0.25, 0.4])

        train_val = np.random.choice(['train', 'test'], 1, p=[0.7, 0.3])[0]

        lbl_dest_path = os.path.join(
            lbl_dst_dir, train_val, "{:06d}.txt".format(i+1))
        lbl_dest_file = open(lbl_dest_path, 'w+')
        tot_lbl = []
        for _ in range(num_pieces[0]):
            # for j in range(len(crops)):

            # Select random crop from all
            num_piece = randint(0, len(crops)-1)
            crop = crops[num_piece]
            crop_path = crops_path[num_piece]
            # crop = crops[j]

            # Get original label
            with open(os.path.join(lbl_og_dir, os.path.basename(crop_path)[:-3]+'txt')) as lbl_og_file:
                # lbl_og = lbl_og_file.readlines()[0].strip().split(' ')
                lbl_og = list(map(
                    float, lbl_og_file.readline().strip().split(' ')))
                lbl_og[0] = int(lbl_og[0])

            # Crop 2 gray
            gray_img = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            # Get mask for background
            _, mask = cv2.threshold(gray_img, 60, 255, cv2.THRESH_BINARY)
            # Mask for the crop
            mask_inv = cv2.bitwise_not(mask)

            img_fg = cv2.bitwise_and(crop, crop, mask=mask_inv)

            # Move the crop
            mov = np.random.choice(np.arange(0, 3), 1)
            if mov[0] == 0:
                img_mov, mask_mov, lbl_mov = move_down(
                    img_fg, mask, lbl_og, 100)
            elif mov[0] == 1:
                img_mov, mask_mov, lbl_mov = move_up(img_fg, mask, lbl_og, 100)
            else:
                img_mov = deepcopy(img_fg)
                mask_mov = deepcopy(mask)
                lbl_mov = lbl_og

            mov = np.random.choice(np.arange(0, 3), 1)
            if mov[0] == 0:
                img_mov, mask_mov, lbl_mov = move_right(
                    img_mov, mask_mov, lbl_mov, 100)
            elif mov[0] == 1:
                img_mov, mask_mov, lbl_mov = move_left(
                    img_mov, mask_mov, lbl_mov, 100)

            lbl_dest_file.write("{ob_class} {x_coord} {y_coord} {width} {height}\n".format(
                ob_class=0, x_coord=lbl_mov[1], y_coord=lbl_mov[2], width=lbl_mov[3], height=lbl_mov[4]))

            col_left = (lbl_mov[1]-lbl_mov[3]/2)*cols
            row_up = (lbl_mov[2]-lbl_mov[4]/2)*rows
            width = lbl_mov[3]*cols
            height = lbl_mov[4]*rows
            if col_left < 0:
                width += col_left
                col_left = 0
            if row_up < 0:
                height += row_up
                row_up = 0

            tot_lbl.append(
                (int(col_left), int(row_up), int(width), int(height)))

            base_img_bg_mov = cv2.bitwise_and(roi, roi, mask=mask_mov)
            dst_mov = cv2.add(base_img_bg_mov, img_mov)
            base_img[Y_MIN:rows+Y_MIN, X_MIN:cols+X_MIN] = dst_mov
            bbox_img = draw_bboxes(base_img, tot_lbl, ["G1_a"])

            if show:
                # cv2.imshow('mask', mask)
                # cv2.imshow('mask_mov', mask_mov)
                # cv2.imshow('base_img_mov', base_img_bg_mov)
                # cv2.imshow('img_fg', img_fg)
                # cv2.imshow('img_mov', img_mov)
                cv2.imshow('base_img', base_img)
                cv2.imshow('bbox_img', bbox_img)

                while True:
                    k = cv2.waitKey(1)
                    if k % 256 == 32:
                        # SPACE pressed
                        cv2.destroyAllWindows()

                        break
                    elif k % 256 == 27:
                        cv2.destroyAllWindows()
                        del base_img
                        sys.exit()

        if save:
            cv2.imwrite(os.path.join(
                imgs_dst_dir, train_val, '{:06d}.png'.format(i+1)), base_img)
            cv2.imwrite(os.path.join(
                bbox_dst_dir, train_val, '{:06d}.png'.format(i+1)), bbox_img)

        lbl_dest_file.close()
        added_imgs.append(base_img)
