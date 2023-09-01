import cv2
import numpy as np
import os
import copy
from array import array
from fix_crop import fix_crop
import sys


def get_bbox(mask):
    row_up = None
    row_down = None
    col_left = None
    col_right = None
    dim = np.shape(mask)
    for i in range(0, dim[0]):
        if np.where(mask[i, :] > 0)[0].size > 0 and row_up == None:
            row_up = i
        if i != dim[0]-1:
            if row_up != None and np.where(mask[i, :] > 0)[0].size > 0 and np.where(mask[i+1, :] > 0)[0].size == 0:
                candidato_down = i
                if candidato_down - row_up > 100:
                    row_down = candidato_down
                    break
                else:
                    candidato_down = None
        else:
            row_down = i

    for i in range(0, dim[1]):
        if np.where(mask[0:, i] > 0)[0].size > 0 and col_left == None:
            col_left = i
        if i != dim[1]-1:
            if col_left != None and np.where(mask[0:, i] > 0)[0].size > 0 and np.where(mask[0:, i+1] > 0)[0].size == 0:
                candidato_right = i
                if candidato_right - col_left > 100:
                    col_right = candidato_right
                    break
                else:
                    candidato_right = None

        else:
            col_right = i
    return row_up, row_down, col_left, col_right


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
    bboxes_image = copy.deepcopy(image)
    for i in range(len(categories)):
        category = categories[i]
        # Change of format: x1,y1,h,w -> x1,y1,x2,y2
        bbox = [bboxes[i][0], bboxes[i][1],
                (bboxes[i][0] + bboxes[i][2]), (bboxes[i][1] + bboxes[i][3])]
        cv2.rectangle(
            bboxes_image, (bbox[0], bbox[1]),
            (bbox[2], bbox[3]), (0, 0, 255), 2)
        cv2.rectangle(
            bboxes_image, (bbox[0], bbox[1]-30),
            (bbox[0]+80, bbox[1]), (0, 0, 255), -1)
        cv2.putText(bboxes_image, category,
                    (bbox[0], bbox[1]-5), 0, 1, (255, 255, 255), 2)
    return bboxes_image


if __name__ == "__main__":

    fc = fix_crop()
    for file in os.listdir('./datasets/G1_a_real_360/images'):

        cam = cv2.VideoCapture(0)
        ret, frame = cam.read()

        image_base = cv2.imread("./fotos_base/base_new.png")
        dim_img_base = np.shape(image_base)

        show = True
        new_frame = copy.deepcopy(frame)
        img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        dim_frame = np.shape(frame)
        lower_bound = 0
        upper_bound = 90
        mask = cv2.inRange(img_gray, lower_bound, upper_bound)
        for i in range(dim_frame[0]):
            for j in range(dim_frame[1]):
                if (mask[i, j] == 0).all():
                    new_frame[i, j] = image_base[i, j]
        cv2.imshow('mask', mask)
        cv2.imshow('frame', frame)
        while True:
            k = cv2.waitKey(1)
            if k % 256 == 32:
                # SPACE pressed
                cv2.destroyAllWindows()

                break
            elif k % 256 == 27:
                cv2.destroyAllWindows()
                del frame
                sys.exit()
        row_up, row_down, col_left, col_right = get_bbox(mask)

        width = col_right-col_left
        height = row_down-row_up
        print("first draft:", col_left, row_up, width, height)

        [col_left, row_up, width, height] = fc.update_crop_dims(
            [col_left, row_up, width, height])
        print("second draft: ", col_left, row_up, width, height)

        bbox_im = draw_bboxes(
            new_frame, [(col_left, row_up, width, height)], ["G1_a"])

        crop = fc.extract_crop(
            new_frame, [col_left, row_up, width, height])
        crop = cv2.resize(crop, [640, 640])
        cv2.imshow('frame', frame)
        cv2.imshow('gray', img_gray)
        cv2.imshow('mask', mask)
        cv2.imshow('bbox', bbox_im)
        cv2.imshow('crop', crop)
        cv2.imshow('new', new_frame)
        # cv2.imshow('diff2', Conv_hsv_Gray)
        # cv2.imshow('mask_0', mask_0)

        # cv2.imshow('result', result)
        while True:
            k = cv2.waitKey(1)
            if k % 256 == 32:
                # SPACE pressed
                cv2.destroyAllWindows()
                del frame
                break
            elif k % 256 == 27:
                cv2.destroyAllWindows()
                del frame
                sys.exit()
            elif k % 256 == 112:
                cv2.imwrite('pic_filtered.png', new_frame)
