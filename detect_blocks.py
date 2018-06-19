import cv2
import numpy as np
import datetime


def detect_objects(block_number):
    # take picture from webcam
    now = datetime.datetime.now()
    file_name = now.strftime("%Y-%m-%d")
    img = cv2.imread(str(take_picture(file_name)), 1)
    # img = cv2.imread("/home/Documents/2018-01-27.png", 1)
    # TODO Might change imgHSV to imgRGB and correspondingly modify findObjectWithinRange
    imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # TODO three colors of blocks, red green blue, MEASURE threshold
    red_low = np.array([255, 0, 0], dtype=np.uint8)
    red_high = np.array([255, 0, 0], dtype=np.uint8)
    red = [255, 0, 0]

    green_low = np.array([255, 0, 0], dtype=np.uint8)
    green_high = np.array([255, 0, 0], dtype=np.uint8)
    green = [255, 0, 0]

    blue_low = np.array([255, 0, 0], dtype=np.uint8)
    blue_high = np.array([255, 0, 0], dtype=np.uint8)
    blue = [255, 0, 0]


    def red_block():
        # TODO Get the centers of found red blocks in two-dimensional list
        center = find_element_within_range(img, imgHSV, red_low, red_high, red)
        # TODO Implement recognize_block method, it returns the Block NUMBER found
        # TODO  and the center of it in a two-dimensional list
        block_info = recognize_block(img, imgHSV, block_number, center)
        number = block_info[1]
        block_location = block_info[2]
        return number, block_location

    def green_block():
        center = find_element_within_range(img, imgHSV, green_low, green_high, green)

        block_info = recognize_block(img, imgHSV, block_number, center)
        number = block_info[1]
        block_location = block_info[2]

        return number, block_location

    def blue_block():
        center = find_element_within_range(img, imgHSV, blue_low, blue_high, blue)

        block_info = recognize_block(img, imgHSV, block_number, center)
        number = block_info[1]
        block_location = block_info[2]

        return number, block_location

    # map the inputs to the function blocks
    options = {1: red_block(),
               2: red_block(),
               3: red_block(),
               4: green_block(),
               5: green_block(),
               6: green_block(),
               7: blue_block(),
               8: blue_block(),
               9: blue_block()}

    b_number, rec_block_location = options[block_number]

    if block_number == b_number:
        print "You asked for block number ", block_number
    elif b_number == 0:
        print "Could not detect a block, abort."
    else:
        print "Block recognized to be ", b_number, " while you asked for block ", block_number

    cv2.imshow('image', img)
    cv2.waitKey(0)

    cv2.destroyAllWindows()

    return rec_block_location

def recognize_block(img, imgHSV, block_number, center):

    return 0


def find_element_within_range(image, imgHSV, lower_range, upper_range, color):
    mask = cv2.inRange(imgHSV, lower_range, upper_range)

    cv2.imshow('mask', mask)
    cv2.waitKey(0)

    element = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
    mask = cv2.erode(mask, element, iterations=2)
    mask = cv2.dilate(mask, element, iterations=2)
    mask = cv2.erode(mask, element)

    _, contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    maximumArea = 0
    bestContour = None
    for contour in contours:
        currentArea = cv2.contourArea(contour)
        if currentArea > maximumArea:
            bestContour = contour
            maximumArea = currentArea
    # Create a bounding box around the biggest red object
    x, y, w, h = (0, 0, 0, 0)

    if bestContour is not None:
        x, y, w, h = cv2.boundingRect(bestContour)
        cv2.rectangle(image, (x, y), (x + w, y + h), color, 3)

    if x != 0:
        cv2.circle(image, (x + w / 2, y + h / 2), 3, 3)
        center = (x + w / 2, y + h / 2)
    else:
        center = 0

    return center


def take_picture(file_name):
    # Camera 0 is the camera on the arm
    camera_port = 0

    # Number of frames to throw away while the camera adjusts to light levels
    ramp_frames = 30

    # Now we can initialize the camera capture object with the cv2.VideoCapture class.
    # All it needs is the index to a camera port.
    camera = cv2.VideoCapture(camera_port)

    # Captures a single image from the camera and returns it in PIL format
    def get_image():
        # read is the easiest way to get a full image out of a VideoCapture object.
        retval, im = camera.read()
        return im

    # Ramp the camera - these frames will be discarded and are only used to allow v4l2
    # to adjust light levels, if necessary
    for i in xrange(ramp_frames):
        temp = get_image()
    print("Taking image...")
    # Take the actual image we want to keep
    camera_capture = get_image()
    print("Done")
    path_to_file = "/home/team18/image-rec-color/" + str(file_name) + ".png"
    print path_to_file
    # A nice feature of the imwrite method is that it will automatically choose the
    # correct format based on the file extension you provide. Convenient!
    cv2.imwrite(path_to_file, camera_capture)

    # # You'll want to release the camera, otherwise you won't be able to create a new
    # # capture object until your script exits
    camera.release()
    return path_to_file

    # detect_objects(1)
