import os
import numpy as np


if __name__ == "__main__":
    base_dir = os.getcwd()
    dataset_dir = os.path.join(base_dir, "datasets", "G1_a_sin")
    img_dir = os.path.join(dataset_dir, "images")
    lbl_dir = os.path.join(dataset_dir, "labels")
    if not os.path.exists(os.path.join(img_dir, "train")):
        os.makedirs(os.path.join(img_dir, "train"))
    if not os.path.exists(os.path.join(img_dir, "test")):
        os.makedirs(os.path.join(img_dir, "test"))
    if not os.path.exists(os.path.join(lbl_dir, "train")):
        os.makedirs(os.path.join(lbl_dir, "train"))
    if not os.path.exists(os.path.join(lbl_dir, "test")):
        os.makedirs(os.path.join(lbl_dir, "test"))

    for img_name, txt_name in zip(os.listdir(img_dir), os.listdir(lbl_dir)):
        choice = np.random.choice(2, 1, p=[0.3, 0.7])
        if choice:
            os.replace(os.path.join(img_dir, img_name), os.path.join(img_dir, "train", img_name))
            os.replace(os.path.join(lbl_dir, txt_name), os.path.join(lbl_dir, "train", txt_name))
        else:
            os.replace(os.path.join(img_dir, img_name), os.path.join(img_dir, "test", img_name))
            os.replace(os.path.join(lbl_dir, txt_name), os.path.join(lbl_dir, "test", txt_name))
