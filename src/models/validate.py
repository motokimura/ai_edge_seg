#!/usr/bin/env python
import argparse
import os

import numpy as np
from PIL import Image
from skimage import io
from tqdm import tqdm

from predict import SegmentationModel


if __name__ == '__main__':

	parser = argparse.ArgumentParser(description='Validate SS model')
	parser.add_argument('model', help='Path to model weight file')
	parser.add_argument('outdir', help='Output directory')
	parser.add_argument('--base-width', '-bw', type=int, default=32,
						help='Base width of U-Net')
	parser.add_argument('--scale', '-s', type=float, default=0.5)
	parser.add_argument('--noclahe', dest='clahe', action='store_false')
	parser.add_argument('--root', default='../../data/aiedge')
	parser.add_argument('--gpu', '-g', type=int, default=0)
	args = parser.parse_args()
	
	mask_dir = os.path.join(args.outdir, 'val_mask')
	os.makedirs(mask_dir)

	score_dir = os.path.join(args.outdir, 'val_score')
	os.makedirs(score_dir)

	mean_path = os.path.join(args.root, 'mean.npy')
	mean = np.load(mean_path)

	model = SegmentationModel(
		args.model, mean, 
		scale=args.scale, clahe=args.clahe, gpu=args.gpu, base_width=args.base_width, class_weight=None
	)

	val_list = os.path.join(args.root, 'val.txt')
	with open(val_list, 'r') as f:
		lines = f.readlines()
	
	for line in tqdm(lines):
		line = line.rstrip()
		image_rel_path, _ = line.split(',')
		image_path = os.path.join(args.root, image_rel_path)
		pil_image = Image.open(image_path)

		score, mask = model.apply_segmentation(pil_image)

		image_file = os.path.basename(image_rel_path)
		basename, _ = os.path.splitext(image_file)
		mask_path = os.path.join(mask_dir, '{}.png'.format(basename))
		io.imsave(mask_path, mask)
		score_path = os.path.join(score_dir, '{}.npy'.format(basename))
		np.save(score_path, score)