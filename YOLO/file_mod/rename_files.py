import os
import numpy as np

base_dir = os.getcwd()
datasets_dir = os.path.join(os.getcwd(), 'datasets')

og_dir = os.path.join(datasets_dir,
                    #   'G1_a_regressor_saliency',
                      'datasetVirtualLu',
                      'images')


def rename_files(dir, offset=0):
    for i, file in enumerate(os.listdir(dir)):
        new_img_name = "{:06d}.png".format(i+offset)
        os.rename(os.path.join(dir, file),
                  os.path.join(dir, new_img_name))


def move_files(og_dir):
    for file in os.listdir(og_dir):
        if not os.path.exists(os.path.join(og_dir, 'train')):
            os.makedirs(os.path.join(og_dir, 'train'))
        if not os.path.exists(os.path.join(og_dir, 'test')):
            os.makedirs(os.path.join(og_dir, 'test'))
        train_val = np.random.choice(['train', 'test'], 1, p=[0.8, 0.2])[0]
        new_dir = os.path.join(og_dir, train_val)
        os.rename(os.path.join(og_dir, file),
                  os.path.join(new_dir, file))


if __name__ == "__main__":
    

    move_files(og_dir)
