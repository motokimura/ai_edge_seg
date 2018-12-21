#!/usr/bin/env python
import argparse
import os
import numpy as np

import sys
sys.path.append("../models")

from dataset import LabeledImageDataset


def compute_mean(dataset):
	print('Computing mean image...')

	sum_rgb = np.zeros(shape=[3, ])
	N = len(dataset)
	for i, (image, _) in enumerate(dataset):
		sum_rgb += image.mean(axis=(1, 2), keepdims=False)
		sys.stderr.write('{} / {}\r'.format(i, N))
		sys.stderr.flush()
	sys.stderr.write('\n')
	mean = sum_rgb / N

	print("Done!")
	print("Computed mean: (R, G, B) = ({}, {}, {})".format(mean[0], mean[1], mean[2]))

	return mean


if __name__ == '__main__':

	parser = argparse.ArgumentParser(description='Compute images mean array')
	parser.add_argument('data_type', choices=['cityscapes', 'aiedge'])
	args = parser.parse_args()

	if args.data_type == 'cityscapes':
		data_root = '../../data/cityscapes'
		crop_wh = (2048, 1024)
	if args.data_type == 'aiedge':
		# TBI
		pass

	data_list = os.path.join(data_root, 'train.txt')
	output = os.path.join(data_root, 'mean.npy')

	dataset = LabeledImageDataset(args.data_type, data_list, data_root, crop_wh)
	mean = compute_mean(dataset)

	np.save(output, mean)