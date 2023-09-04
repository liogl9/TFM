import os
import cv2

def create_video_from_images(input_dir, output_file, fps):
    # Get the list of PNG images in the input directory
    image_files = sorted([f for f in os.listdir(input_dir) if f.endswith('.png')])

    # Read the first image to get the dimensions
    image_path = os.path.join(input_dir, image_files[0])
    first_image = cv2.imread(image_path)
    height, width, _ = first_image.shape

    # Create a video writer object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_file, fourcc, fps, (width, height))

    for image_file in image_files:
        image_path = os.path.join(input_dir, image_file)
        image = cv2.imread(image_path)

        # Write the image to the video writer
        video_writer.write(image)

    # Release the video writer
    video_writer.release()

# Example usage
input_directory = os.path.join(os.getcwd(), "GanResults","inference","0-360_z_10_30-x_white_fake2real")
output_filename = '10_30grados.mp4'
frame_rate = 24

create_video_from_images(input_directory, output_filename, frame_rate)