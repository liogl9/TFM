import cv2
import numpy as np
import os
import copy
from array import array
import time
from fix_crop import fix_crop


def get_bbox(frame):
    """
    get_bbox _summary_

    _extended_summary_

    Args:
        frame (_type_): _description_

    Returns:
        _type_: _description_
    """
    row_up = None
    row_down = None
    col_left = None
    col_right = None

    # Threshold of black in RGB space
    lower_black = np.array([0, 0, 0])
    upper_black = np.array([50, 50, 50])

    # preparing the mask to overlay
    mask = cv2.inRange(frame, lower_black, upper_black)
    dim = np.shape(mask)
    for i in range(6, dim[0]):
        if np.where(mask[i, :] > 0)[0].size > 0 and row_up == None:
            row_up = i
        if i != dim[0]-1:
            if row_up != None and np.where(mask[i, :] > 0)[0].size > 0 and np.where(mask[i+1, :] > 0)[0].size == 0:
                row_down = i
                if row_down - row_up > 200:
                    break
                else:
                    row_up = None
                    row_down = None
        else:
            row_down = i

    for i in range(0, dim[1]):
        if np.where(mask[6:, i] > 0)[0].size > 0 and col_left == None:
            col_left = i
        if i != dim[1]-1:
            if col_left != None and np.where(mask[0:, i] > 0)[0].size > 0 and np.where(mask[0:, i+1] > 0)[0].size == 0:
                col_right = i
                if col_right - col_left > 200:
                    break
                else:
                    col_right = None
                    col_left = None
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


if __name__ == "__main__":

    base_dir = os.getcwd()
    img_og_dir = os.path.join(base_dir, "datasets",
                              "G1_a_real_20_oct", "images")
    img_new_dir = os.path.join(base_dir, "datasets",
                               "G1_a_real_20_oct", "images_regressor_224")
    if not os.path.exists(img_new_dir):
        os.makedirs(img_new_dir)
    show = False
    fc = fix_crop()
    for img_name in os.listdir(img_og_dir):
        frame = cv2.imread(os.path.join(img_og_dir, img_name))

        dim = np.shape(frame)

        dim_frame = np.shape(frame)

        # Threshold of black in RGB space

        # Masked img
        image_base = cv2.imread("./fotos_base/base_new.png")
        dim_img_base = np.shape(image_base)
        new_frame = copy.deepcopy(frame)
        img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        lower_bound = 0
        upper_bound = 90
        mask = cv2.inRange(img_gray, lower_bound, upper_bound)
        for i in range(dim_frame[0]):
            for j in range(dim_frame[1]):
                if (mask[i, j] == 0).all():
                    new_frame[i, j] = image_base[i, j]

        # Saco Boundig box con color filtering
        row_up, row_down, col_left, col_right = get_bbox(new_frame)
        if col_left == None or row_up == None:
            continue
        width = col_right-col_left
        height = row_down-row_up
        center = [col_left + (width // 2),
                  row_up + (height // 2)]
        print("first draft: ", col_left, row_up, width, height)
        padding_need = 0
        if col_left == 0 or row_up == 0 or col_left + width > 640 or row_up+height > 480:
            padding_need = 1
        # Hago update para crop del regresor
        [col_left, row_up, width, height] = fc.update_crop_dims(
            [col_left, row_up, width, height])

        # Creo nueva imagen con bounding box pintada
        bbox_img = draw_bboxes(
            new_frame, [(col_left, row_up, width, height)], ["G1_a"])

        print("second draft: ", col_left, row_up, width, height)

        crop_img = fc.extract_crop(
            new_frame, [col_left, row_up, width, height])

        crop_img = cv2.resize(crop_img, [224, 224])

        # Obtengo normales
        if show:
            cv2.namedWindow("test")
            cv2.namedWindow("mask")
            cv2.namedWindow("crop")
            init_time = time.time()
            while time.time()-init_time < 3:
                cv2.imshow("test", new_frame)
                cv2.imshow("mask", mask)
                cv2.imshow("crop", crop_img)

                k = cv2.waitKey(1)
                if k % 256 == 27:
                    # ESC pressed
                    print("Escape hit, closing...")
                    break

            cv2.destroyAllWindows()
        # Guardo foto original
        og_img_path = os.path.join(img_new_dir, img_name)
        cv2.imwrite(og_img_path, crop_img)
