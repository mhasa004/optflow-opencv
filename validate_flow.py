import os
import sys
import numpy as np
import cv2 as cv


def main(video_path):
	cap = cv.VideoCapture(video_path)
	while cap.isOpened():
		ret, frame = cap.read()
		if not ret:
			break
		print(np.sum(frame[:,:,2]))
	
	cap.release()


if __name__ == '__main__':
	if len(sys.argv) < 2:
		print('Please provide valid input video.')
		exit(0)

	main(sys.argv[1].strip())