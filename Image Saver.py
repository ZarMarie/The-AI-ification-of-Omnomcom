from cv2 import VideoCapture, imshow, imwrite, waitKey, destroyWindow

cam_port = 1
cam = VideoCapture(cam_port)


num = 1
while True:
    result, image = cam.read()

    if result:

        # showing result, it take frame name and image
        # output
        imshow("FoodItem", image)

        k = waitKey(5)

        if k == ord('q'):
            destroyWindow("FoodItem")
            break

        if k == ord(' '):
            # saving image in local storage
            imwrite(f"AppleJuice/{num}.png", image)
            num += 1

    # If captured image is corrupted, moving to else part
    else:
        print("No image detected. Please! try again")