from scipy.spatial import distance as dist
from imutils import perspective
from imutils import contours
import numpy as np
import argparse
import imutils
import cv2
import time

img_size = 60 # has to be the same as for training
pad = 50
#cap = cv2.VideoCapture('video1.mp4')


def midpoint(ptA, ptB):
    return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)


def bbox(img, c):
    x, y, w, h = cv2.boundingRect(c)
    return img[y - pad:y + h + pad, x - pad:w + x + pad], (x, y)


from keras.models import load_model

model = load_model('shapesmodel.h5')
dimData = np.prod([img_size, img_size])

while True:
    img = cv2.imread("objects1.jpg")
    #_, img = cap.read()
    imgc = img.copy()
    # image = cv2.resize(image, (640, 480))
    h, w, c = img.shape
    print(h, w)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    # perform edge detection, then perform a dilation + erosion to
    # close gaps in between object edges
    edged = cv2.Canny(gray, 50, 100)
    edged = cv2.dilate(edged, None, iterations=1)
    edged = cv2.erode(edged, None, iterations=1)
    # find contours in the edge map
    cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    # sort the contours from left-to-right and initialize the
    # 'pixels per metric' calibration variable
    (cnts, _) = contours.sort_contours(cnts)
    for c in cnts:
        # if the contour is too big or too small, it can be ignored
        area = cv2.contourArea(c)
        # print area
        if area > 500 and area < 1180000:

            # crop out the green shape
            roi, coords = bbox(img, c)

            # filter out contours that are long and stringy
            if np.prod(roi.shape[:2]) > 10:

                # get the black and white image of the shape
                roi = cv2.resize(roi, (img_size, img_size))
                roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                (thresh, roi) = cv2.threshold(roi, 225, 255, cv2.THRESH_BINARY)
                #cv2.imshow('roi',cv2.resize(roi, (640, 480)))
                print(roi.shape)

                # feed image into model
                prediction = model.predict(roi.reshape(1, dimData))[0].tolist()

                # create text --> go from categorical labels to the word for the shape.
                text = ''
                p_val, th = .25, .5
                if max(prediction) > p_val:
                    if prediction[0] > p_val and prediction[0] == max(prediction): text, th = 'triangle', prediction[0]
                    if prediction[1] > p_val and prediction[1] == max(prediction): text, th = 'star', prediction[1]
                    if prediction[2] > p_val and prediction[2] == max(prediction): text, th = 'square', prediction[2]
                    if prediction[3] > p_val and prediction[3] == max(prediction): text, th = 'circle', prediction[3]

                # draw the contour
                cv2.drawContours(imgc, c, -1, (0, 0, 255), 2)

                # draw the text
                org, font, color = (coords[0], coords[1] + int(area / 400)), cv2.FONT_HERSHEY_SIMPLEX, (0, 0, 255)
                cv2.putText(imgc, text, org, font, 2, color, int(6 * th), cv2.LINE_AA)


    cv2.imshow('img', cv2.resize(imgc, (640, 480)))  # expect 2 frames per second
    k = cv2.waitKey(10)
    if k == 27: break