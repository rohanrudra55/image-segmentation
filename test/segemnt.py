import cv2 as cv
import numpy as np


def draw_bounding_box(click, x, y, flag_param, parameters):
    global x_pt, y_pt, drawing, top_left_point, bottom_right_point, original_image

    if click == cv.EVENT_LBUTTONDOWN:
        """
        It will activate when left button will be pushed
            x_pt : Storing initial X
            y_pt : Storing initial Y
            drawing : To mark the work is done
        """
        drawing = True
        x_pt, y_pt = x, y

    elif click == cv.EVENT_MOUSEMOVE:
        """
        image : to mark the selected area and reducing the process
        """
        if drawing:
            top_left_point, bottom_right_point = (x_pt, y_pt), (x, y)
            image[y_pt:y, x_pt:x] = 255 - original_image[y_pt:y, x_pt:x]
            cv.rectangle(image, top_left_point,
                         bottom_right_point, (0, 255, 0), 2)

    elif click == cv.EVENT_LBUTTONUP:
        drawing = False
        top_left_point, bottom_right_point = (x_pt, y_pt), (x, y)
        image[y_pt:y, x_pt:x] = 255 - image[y_pt:y, x_pt:x]
        cv.rectangle(image, top_left_point,
                     bottom_right_point, (0, 255, 0), 2)
        bounding_box = (x_pt, y_pt, x-x_pt, y-y_pt)

        grabcut_algorithm(original_image, bounding_box)


def grabcut_algorithm(original_image, bounding_box):
    """
    User inputs the rectangle. Everything outside 
    this rectangle will be taken as sure background. 
    And inside as forground. then in an itterative process 
    it will chose the probable correct subjct.
    bounding_box : instead of initializing in rect mode, 
                    you can directly go into mask mode
    """

    segment_mask = np.zeros(original_image.shape[:2], np.uint8)

    x, y, width, height = bounding_box
    segment_mask[y:y+height, x:x+width] = cv.GC_FGD

    background_mdl = np.zeros((1, 65), np.float64)
    foreground_mdl = np.zeros((1, 65), np.float64)
    interation_count = 5

    cv.grabCut(original_image, segment_mask, bounding_box, background_mdl, foreground_mdl,
               interation_count, cv.GC_INIT_WITH_RECT)

    new_mask = np.where((segment_mask == 2) | (segment_mask == 0), 0, 1).astype('uint8')

    original_image = original_image*new_mask[:, :, np.newaxis]
    print(original_image)

    cv.imshow('Result', original_image)


if __name__ == '__main__':
    drawing = False
    top_left_point, bottom_right_point = (-1, -1), (-1, -1)

    original_image = cv.imread("test.jpg")
    original_image = cv.resize(original_image, (500, 500))
    image = original_image.copy()
    cv.namedWindow('Frame')
    # For manual inputs
    # bounding_box = (50, 50, 300, 300)
    # grabcut_algorithm(image, bounding_box)
    cv.setMouseCallback('Frame', draw_bounding_box)

    while True:
        cv.imshow('Frame', image)
        c = cv.waitKey(1)
        if c == 27:
            break

    cv.destroyAllWindows()
