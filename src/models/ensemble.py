#!/usr/bin/env python
import argparse
import os

import numpy as np
from skimage import io
from tqdm import tqdm

if __name__ == '__main__':

	parser = argparse.ArgumentParser(description='Ensemble')
	parser.add_argument('input_1', help='Model-1 score dir')
	parser.add_argument('input_2', help='Model-2 score dir')
	parser.add_argument('outdir', help='Output directory')
	args = parser.parse_args()

	mask_dir = os.path.join(args.outdir, 'mask')
	os.makedirs(mask_dir)

	filenames = os.listdir(args.input_1)

	for filename in tqdm(filenames):
		score_1 = np.load(os.path.join(args.input_1, filename))
		score_2 = np.load(os.path.join(args.input_2, filename))

		h, w, c = score_1.shape
		score = np.zeros(shape=[h, w, c], dtype=score_1.dtype)
		score = (score_1 + score_2) / 2.0

		mask = np.zeros(shape=[h, w, 3], dtype=np.uint8)
		class_idx = np.argmax(score, axis=2)
		mask[class_idx == 0] = np.array([0, 0, 0])      # 0: "background"
		mask[class_idx == 1] = np.array([0, 0, 255])    # 1: "car"
		mask[class_idx == 2] = np.array([255, 0, 0])    # 2: "pedestrian"
		mask[class_idx == 3] = np.array([255, 255, 0])  # 3: "signal"
		mask[class_idx == 4] = np.array([69, 47, 142])  # 4: "lane" (road + parking)

		basename, _ = os.path.splitext(filename)
		out_path = os.path.join(mask_dir, '{}.png'.format(basename))
		io.imsave(out_path, mask)