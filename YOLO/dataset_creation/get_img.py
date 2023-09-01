# import cv2


# # of the points clicked on the image
# def click_event(event, x, y, flags, params):

#     # checking for left mouse clicks
#     if event == cv2.EVENT_LBUTTONDOWN:

#         # displaying the coordinates
#         # on the Shell
#         print(x, ' ', y)

#         # displaying the coordinates
#         # on the image window
#         font = cv2.FONT_HERSHEY_SIMPLEX
#         cv2.putText(frame, str(x) + ',' +
#                     str(y), (x, y), font,
#                     1, (255, 0, 0), 2)
#         cv2.imshow('test', frame)


# # driver function
# if __name__ == "__main__":

#     cam = cv2.VideoCapture(2)

#     cv2.namedWindow("test")

#     img_counter = 0
#     # setting mouse handler for the image
#     # and calling the click_event() function
#     cv2.setMouseCallback('test', click_event)

#     while True:
#         ret, frame = cam.read()
#         if not ret:
#             print("failed to grab frame")
#             break
#         cv2.imshow("test", frame)

#         k = cv2.waitKey(1)
#         if k % 256 == 27:
#             # ESC pressed
#             print("Escape hit, closing...")
#             break

#     cam.release()

#     cv2.destroyAllWindows()
import cv2

cam = cv2.VideoCapture(2)

cv2.namedWindow("test")
ret, frame = cam.read()
img_counter = 0

while True:
    ret, frame = cam.read()
    # frame = cv2.line(frame, (0, 100), (640, 100), (0, 0, 0), 2)

    if not ret:
        print("failed to grab frame")
        break
    cv2.imshow("test", frame)

    k = cv2.waitKey(1)
    if k % 256 == 27:
        # ESC pressed
        print("Escape hit, closing...")
        break
    elif k % 256 == 32:
        # SPACE pressed
        img_name = "base_{}.png".format(img_counter)
        cv2.imwrite(img_name, frame)
        print("{} written!".format(img_name))
        img_counter += 1

cam.release()

cv2.destroyAllWindows()
