import os
import sys
import numpy as np
import glob
from subprocess import call


def main(input_dir, output_dir):
	if not os.path.exists(output_dir):
		os.makedirs(output_dir)

	videos = glob.glob(input_dir + '/*')
	for video in videos:
		video_name = os.path.basename(video)
		flow_path = os.path.join(output_dir, video_name)
		cmd = './mainflow.out %s %s\n' % (video, flow_path)
		print(cmd)
		# call(cmd, shell=True)


if __name__ == '__main__':
	if len(sys.argv) < 3:
		print('Please provide input and output dirs')
		exit(0)

	main(sys.argv[1].strip(), sys.argv[2].strip())