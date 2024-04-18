from cv2 import VideoCapture, imshow, imwrite, waitKey, destroyWindow

# Camera port
cam_port = 1
cam = VideoCapture(cam_port)

# Set image number manually here
num = 1
while True:
    # get camera capture
    result, image = cam.read()

    if result:

        # Show the result of the camera capture
        imshow("FoodItem", image)

        # wait 5ms for key press so camera feed stays live
        k = waitKey(5)

        # if q is pressed, the program is closed
        if k == ord('q'):
            destroyWindow("FoodItem")
            break
        # if space is pressed, the image is saved under the current number (currently in apple juice folder, but
        # changed manually), and the number is incremented
        if k == ord(' '):
            imwrite(f"AppleJuice/{num}.png", image)
            num += 1

    # If captured image is corrupted or not detected, an error message is printed
    else:
        print("No image detected. Please! try again")