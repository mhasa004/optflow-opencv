import os
import sys
import numpy as np
import cv2 as cv


def main(video_path):
    cap = cv.VideoCapture(video_path)
    optical_flow = cv.DualTVL1OpticalFlow_create()
    ret, frame0 = cap.read()
    frame0 = cv.cvtColor(frame0, cv.COLOR_BGR2GRAY)
    while cap.isOpened():
        ret, frame1 = cap.read()
        if not ret:
            break
        frame1 = cv.cvtColor(frame1, cv.COLOR_BGR2GRAY)
        flow = optical_flow.calc(frame0, frame1, None)
        frame0 = frame1

        print(flow.shape)

    cap.release()


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print('Please provide valid input video and output path.')
        exit(0)

    main(sys.argv[1].strip())
