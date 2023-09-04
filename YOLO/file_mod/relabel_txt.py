import os


# Using readlines()
og_dir = os.path.join(os.getcwd(),'datasets',
                    #   'G1_a_regressor_saliency'
                      'Regressor_Real', 'labels'
                      )
file_path = os.path.join(og_dir,'G1_a_Point_2.txt')
new_file_path = os.path.join(og_dir, 'new_labels.txt')
offset = 0
with open(file_path, 'r') as file1:
    Lines = file1.readlines()


# Strips the newline character
for i, line in enumerate(Lines):
    if i == 0: 
        continue
    line_parts = line.split(' ')
    line_parts[0] = '{:06d}.png'.format(i-1+offset)
    new_line = ' '.join(line_parts)
    new_file = open(new_file_path,'+a')
    new_file.write(new_line)
