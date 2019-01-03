#!/usr/bin/env python
import argparse
import os

import numpy as np
from skimage import io
from tqdm import tqdm

if __name__ == '__main__':

	parser = argparse.ArgumentParser(description='Ensemble')
	parser.add_argument('--inputs', '-i', nargs='+', help='Model score directories')
	parser.add_argument('--outdir', '-o', default='.', help='Output directory')
	args = parser.parse_args()

	mask_dir = os.path.join(args.outdir, 'mask')
	os.makedirs(mask_dir)

	filenames = os.listdir(args.inputs[0])

	for filename in tqdm(filenames):
		score = np.load(os.path.join(args.inputs[0], filename))
		for model in args.inputs[1:]:
			score += np.load(os.path.join(model, filename))
		
		score /= float(len(args.inputs))

		h, w, c = score.shape
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