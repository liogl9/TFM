import cv2
import sys
import torch

frame = cv2.imread('hist_og.png')
while True:
    cv2.imshow("test", frame)

    k = cv2.waitKey(1)
    if k % 256 == 27:
        # ESC pressed
        print("Escape hit, closing...")
        break
    elif k % 256 == 113:
        sys.exit()

cv2.destroyAllWindows()
