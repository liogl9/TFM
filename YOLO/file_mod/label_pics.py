from math import sin, cos
import cv2
import os
import copy
from array import array
import numpy as np
from random import random
from math import sqrt, pi
import keyboard
from examples import rtde_config as rtde_config
from examples import rtde as rtde
import time
import logging
import sys
from examples.control_joint_angles_lio import compute_error, compute_control_effort, list_to_degrees, setp_to_list, list_to_setp
from fix_crop import fix_crop
sys.path.append('..')

OUT_IMG_SIZE = 640
show = False
save = True


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


def euler2normal(euler_x, euler_y, euler_z):
    a_x = sin(euler_z)*sin(euler_x)+cos(euler_z)*sin(euler_y)*cos(euler_x)
    a_y = -cos(euler_z)*sin(euler_x)+sin(euler_z)*sin(euler_y)*cos(euler_x)
    a_z = cos(euler_y)*cos(euler_x)
    return [a_x, a_y, a_z]


def list_deg2rad(angles):

    return [i*pi/180 for i in angles]
    while not (keyboard.is_pressed("q")):

        # Receive UR Robot state
        state = con.receive()

        # If state is None break loop and end connection
        if state is None:
            print("state = NONe")
            break

        # Check if the program is running in the Polyscope
        if state.output_int_register_0 != 0:
            if (time.time() - init_time) >= 1:
                init_time = time.time()
            # compute new target joint angles
            # Compute control error
            error = compute_error(target_joints, state.actual_q)
            # Compute control effort
            control_effort = compute_control_effort(error, gain)
            # Reformat control effort list into setpoint
            list_to_setp(setp, control_effort)
            # Send new control effort
            con.send(setp)

        # kick watchdog
        con.send(watchdog)

        tol_high = 0.001
        tol_low = -0.001
        out = True
        for item in error:
            if (item < tol_high and item > tol_low):
                out = True
            else:
                out = False
                break
        if out:
            break

    list_to_setp(setp, [0, 0, 0, 0, 0, 0])
    con.send(setp)
    if state != None:
        print("State ok")

    return state


if __name__ == "__main__":

    dataset_dir = os.path.join(os.getcwd(), 'datasets', 'G1_a_real_360')
    img_dir = os.path.join(dataset_dir, 'images')
    bbox_dir = os.path.join(dataset_dir, 'bboxes')
    yolo_lbl_dir = os.path.join(dataset_dir, 'labels')
    regressor_img_dir = os.path.join(dataset_dir, 'images_regressor_minSize')

    if not os.path.exists(dataset_dir):
        os.makedirs(dataset_dir)
        os.mkdir(img_dir)
        os.mkdir(bbox_dir)
        os.mkdir(yolo_lbl_dir)
        os.mkdir(regressor_img_dir)
        current_imgs = 0
        print("Created dataset directories in: ", dataset_dir)
    elif not os.path.exists(regressor_img_dir):
        os.mkdir(regressor_img_dir)
    else:
        current_imgs = len(os.listdir(os.path.join(dataset_dir, 'images')))

    # Save initial time
    init_time = time.time()

    file_pad_path = os.path.join(
        dataset_dir, 'imgs_padd.txt')
    if not os.path.exists(file_pad_path):
        padd_file = open(file_pad_path, 'w+')
        padd_file.write('Name\n')
    else:
        padd_file = open(file_pad_path, 'a')

    # Genero variación random
    # print("Moving to init position")
    # mid_joints = [pi, -pi/8, -pi/2-pi/4, -pi/2-pi/8, -pi/2, 0]
    # state = send_pos_rad(con, mid_joints, watchdog, init_time)
    # print("Rob in init pos")

    fc = fix_crop()

    for i, file in enumerate(os.listdir(img_dir)):

        # Hago foto
        frame = cv2.imread(os.path.join(img_dir, file))
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
        for k in range(dim_frame[0]):
            for j in range(dim_frame[1]):
                if (mask[k, j] == 0).all():
                    new_frame[k, j] = image_base[k, j]

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
            frame, [(col_left, row_up, width, height)], ["G1_a"])

        print("second draft: ", col_left, row_up, width, height)

        crop_img = fc.extract_crop(
            frame, [col_left, row_up, width, height])
        crop_size = crop_img.shape[0]
        print(crop_size)

        if i == 0:
            min_crop_size = crop_size
        elif crop_size < min_crop_size:
            min_crop_size = crop_size
        # crop_img = cv2.resize(crop_img, [min_crop_size, min_crop_size])

        if show:
            cv2.namedWindow("test")
            cv2.namedWindow("mask")
            while True:
                cv2.imshow("test", new_frame)
                cv2.imshow("mask", mask)

                k = cv2.waitKey(1)
                if k % 256 == 27:
                    # ESC pressed
                    print("Escape hit, closing...")
                    break

            cv2.destroyAllWindows()

        if save:
            # Guardo foto bbox
            # bbox_img_path = os.path.join(bbox_dir, file)
            # cv2.imwrite(bbox_img_path, bbox_img)
            # Guardo crop
            crop_img_path = os.path.join(
                regressor_img_dir, file)
            cv2.imwrite(crop_img_path, crop_img)

            # Creo txt para guardar labels YOLO
            # yolo_txt = open(os.path.join(
            #     yolo_lbl_dir, file[:-3]+'txt'), 'w+')
            # yolo_txt.write("{ob_class} {x_coord} {y_coord} {width} {height}\n".format(
            #     ob_class=0, x_coord=center[0]*1.0/dim[1], y_coord=center[1]*1.0/dim[0], width=width*1.0/dim[1], height=height*1.0/dim[0]))
            # yolo_txt.close()

            # if padding_need != 0:
            #     # # Añado normales a lables_regressor
            #     padd_file.write(file+"\n")

        # print("Moving to mid position")
        # state = send_pos_rad(con, mid_joints, watchdog, init_time)
        # print("Rob in mid pos")
    padd_file.close()
    cv2.destroyAllWindows()
    if save:
        for prev_file in os.listdir(regressor_img_dir):
            prev_crop = cv2.imread(os.path.join(
                regressor_img_dir, prev_file))
            prev_crop = cv2.resize(
                prev_crop, [min_crop_size, min_crop_size])
            # Guardo crop
            crop_prev_path = os.path.join(
                regressor_img_dir, prev_file)
            cv2.imwrite(crop_prev_path, prev_crop)
