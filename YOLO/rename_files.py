import os
import numpy as np

base_dir = os.getcwd()
datasets_dir = os.path.join(os.getcwd(), 'datasets')

og_dir = os.path.join(datasets_dir, 'G1_a_real_360',
                      'images_regressor_minSize')
lbl_dir = "./datasets/G1_a_sin/labels"


def rename_files():
    for i, file in enumerate(os.listdir(lbl_dir)):
        new_img_name = "{:06d}.png".format(i)
        new_lbl_name = "{:06d}.txt".format(i)
        os.rename(os.path.join(lbl_dir, file),
                  os.path.join(lbl_dir, new_lbl_name))


def move_files(path):
    for file in os.listdir(path):
        train_val = np.random.choice(['train', 'test'], 1, p=[0.8, 0.2])[0]
        new_path = os.path.join(path, train_val)
        os.rename(os.path.join(path, file),
                  os.path.join(new_path, file))


if __name__ == "__main__":
    if not os.path.exists(os.path.join(og_dir, 'train')):
        os.makedirs(os.path.join(og_dir, 'train'))
    if not os.path.exists(os.path.join(og_dir, 'test')):
        os.makedirs(os.path.join(og_dir, 'test'))

    move_files(og_dir)
