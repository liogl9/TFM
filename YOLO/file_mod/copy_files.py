import os
import numpy as np
import csv
import shutil

datasets_dir = os.path.join(os.getcwd(), 'datasets')

og_dir = os.path.join(datasets_dir,
                      'Regressor_Sintec_0_30_random')

txt_path = os.path.join(og_dir,'labels','G1_a_Point_2.txt')

og_img_dir = os.path.join(datasets_dir,
                      'Regressor_Sintec_0_30_random',
                      'images_og')

new_img_dir = os.path.join(datasets_dir,
                      'Regressor_Sintec_0_30_random',
                      'images')


def rename_files(dir, offset=0):
    for i, file in enumerate(os.listdir(dir)):
        new_img_name = "{:06d}.png".format(i+offset)
        os.rename(os.path.join(dir, file),
                  os.path.join(dir, new_img_name))


def move_files(og_dir):
    for file in os.listdir(og_dir):
        train_val = np.random.choice(['train', 'test'], 1, p=[0.8, 0.2])[0]
        new_dir = os.path.join(og_dir, train_val)
        os.rename(os.path.join(og_dir, file),
                  os.path.join(new_dir, file))
        
def get_img_lbl(txt_path):
    full_labels = []
    with open(txt_path, "r") as f:
        csvreader = csv.reader(f, delimiter=" ")
        header = next(csvreader)
        for row in csvreader:
            full_labels.append(row[0])
    return full_labels

def copy_files(filenames):
    for file in filenames:
        og_file = os.path.join(og_img_dir,file)
        new_file = os.path.join(new_img_dir,file)
        shutil.copy(og_file,new_file)
     


if __name__ == "__main__":
    # if not os.path.exists(os.path.join(og_dir, 'train')):
    #     os.makedirs(os.path.join(og_dir, 'train'))
    # if not os.path.exists(os.path.join(og_dir, 'test')):
    #     os.makedirs(os.path.join(og_dir, 'test'))

    imgs_name = get_img_lbl(txt_path=txt_path)
    copy_files(imgs_name)
    
    
