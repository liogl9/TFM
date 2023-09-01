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


def send_pos_rad(con, target_joints, watchdog, init_time):
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


show = True

ROBOT_HOST = '192.168.56.94'  # ip in settings in the tablet
ROBOT_PORT = 30004
config_filename = './examples/control_loop_configuration.xml'

logging.getLogger().setLevel(logging.INFO)

# configuration files
conf = rtde_config.ConfigFile(config_filename)
state_names, state_types = conf.get_recipe('state')
setp_names, setp_types = conf.get_recipe('setp')
watchdog_names, watchdog_types = conf.get_recipe('watchdog')


if __name__ == "__main__":

    cam = cv2.VideoCapture(0)

    ret, frame = cam.read()

    base_dir = os.getcwd()
    dataset_dir = os.path.join(base_dir, 'datasets', 'Prueba0')
    img_dir = os.path.join(dataset_dir, 'images')
    bbox_dir = os.path.join(dataset_dir, 'bboxes')
    yolo_lbl_dir = os.path.join(dataset_dir, 'labels')
    regressor_lbl_dir = os.path.join(dataset_dir, 'labels_regressor')
    regressor_img_dir = os.path.join(dataset_dir, 'images_regressor')

    if not os.path.exists(dataset_dir):
        os.makedirs(dataset_dir)
        os.mkdir(img_dir)
        os.mkdir(bbox_dir)
        os.mkdir(yolo_lbl_dir)
        os.mkdir(regressor_lbl_dir)
        os.mkdir(regressor_img_dir)
        current_imgs = 0
        print("Created dataset directories in: ", dataset_dir)
    else:
        current_imgs = len(os.listdir(os.path.join(dataset_dir, 'images')))

    num_imgs = 360

    # Conectar al robot
    # connection to the robot
    con = rtde.RTDE(ROBOT_HOST, ROBOT_PORT)
    con.connect()
    connection_state = con.connect()

    # check connection
    while connection_state != 0:
        print(connection_state)
        time.sleep(5)
        connection_state = con.connect()
    print("Succesful connection to the robot")

    # get controller version
    con.get_controller_version()

    # setup recipes
    con.send_output_setup(state_names, state_types)
    setp = con.send_input_setup(setp_names, setp_types)
    watchdog = con.send_input_setup(watchdog_names, watchdog_types)

    # Initialize 6 registers which will hold the target angle values
    setp.input_double_register_0 = 0.0
    setp.input_double_register_1 = 0.0
    setp.input_double_register_2 = 0.0
    setp.input_double_register_3 = 0.0
    setp.input_double_register_4 = 0.0
    setp.input_double_register_5 = 0.0

    # The function "rtde_set_watchdog" in the "rtde_control_loop.urp" creates a 1 Hz watchdog
    watchdog.input_int_register_0 = 0

    # Start data synchronization
    if not con.send_start():
        sys.exit()

    # Save initial time
    init_time = time.time()
    # Set gain for the control
    gain = 1

    regressor_lbl_path = os.path.join(
        regressor_lbl_dir, 'Labels_regresor.txt')
    # Creo txt para guardar labels regresor
    if not os.path.exists(regressor_lbl_path):
        regressor_file = open(regressor_lbl_path, 'w+')
        regressor_file.write('Name, width, height, u, v, w\n')
    else:
        regressor_file = open(regressor_lbl_path, 'a')

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

    for num_img in range(0, 360):

        # Genero variación random
        # print("Moving to mid position")
        # j1 = random()*180-180
        j1 = 0
        # mid_joints = [pi + j1*pi/180, -pi/8, -pi/2-pi/4, -pi/2-pi/8, -pi/2, 0]
        # state = send_pos_rad(con, mid_joints, watchdog, init_time)
        # print("Rob in mid pos")

        print("Moving to pic position")
        # j4 = random()*60-30
        j4 = 0
        # j5 = random()*60-30
        j5 = 0
        j6 = num_img
        random_joints = [pi + j1*pi/180, -pi/8, -pi/2-pi/4, -pi/2-pi/8 +
                         j4*pi/180, -pi/2+j5*pi/180, j6*pi/180]
        state = send_pos_rad(con, random_joints, watchdog, init_time)
        print("Rob in random pos")

        # Hago foto
        ret, frame = cam.read()
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
            frame, [(col_left, row_up, width, height)], ["G1_a"])

        print("second draft: ", col_left, row_up, width, height)

        crop_img = fc.extract_crop(
            frame, [col_left, row_up, width, height])

        crop_img = cv2.resize(crop_img, [OUT_IMG_SIZE, OUT_IMG_SIZE])

        # Obtengo normales
        euler_ang = [state.output_double_register_0,
                     state.output_double_register_1,
                     state.output_double_register_2]
        normal = euler2normal(euler_ang[0], euler_ang[1], euler_ang[2])
        print("Normals: ", normal)

        if show:
            cv2.namedWindow("test")
            cv2.namedWindow("mask")
            init_time = time.time()
            while time.time()-init_time < 3:
                cv2.imshow("test", new_frame)
                cv2.imshow("mask", mask)

                k = cv2.waitKey(1)
                if k % 256 == 27:
                    # ESC pressed
                    print("Escape hit, closing...")
                    break

            cv2.destroyAllWindows()
        # Guardo foto original
        og_img_path = os.path.join(img_dir, "{:06d}.png".format(num_img))
        cv2.imwrite(og_img_path, frame)
        # Guardo foto bbox
        bbox_img_path = os.path.join(bbox_dir, "{:06d}.png".format(num_img))
        cv2.imwrite(bbox_img_path, bbox_img)
        # Guardo crop
        crop_img_path = os.path.join(
            regressor_img_dir, "{:06d}.png".format(num_img))
        cv2.imwrite(crop_img_path, crop_img)

        # Creo txt para guardar labels YOLO
        yolo_txt = open(os.path.join(
            yolo_lbl_dir, "{:06d}.txt".format(num_img)), 'w+')
        yolo_txt.write("{ob_class} {x_coord} {y_coord} {width} {height}\n".format(
            ob_class=0, x_coord=center[0]*1.0/dim[1], y_coord=center[1]*1.0/dim[0], width=width*1.0/dim[1], height=height*1.0/dim[0]))
        yolo_txt.close()

        # # Añado normales a lables_regressor
        regressor_file.write("{:06d}.png {:0.6f} {:0.6f} {:0.6f} {:0.6f} {:0.6f}\n".format(
            num_img, center[0]*1.0/dim[1], center[1]*1.0/dim[0], normal[0], normal[1], normal[2]))

        if padding_need != 0:
            # # Añado normales a lables_regressor
            padd_file.write("{:06d}.png\n".format(num_img))

        print("{}/{} pictures".format(num_img, num_imgs + current_imgs+1))

        # print("Moving to mid position")
        # state = send_pos_rad(con, mid_joints, watchdog, init_time)
        # print("Rob in mid pos")

    regressor_file.close()
    padd_file.close()
    con.send_pause()

    con.disconnect()
    cam.release()
    cv2.destroyAllWindows()
