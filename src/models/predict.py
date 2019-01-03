#!/usr/bin/env python
import argparse
import os

import numpy as np
import cv2
import math
from PIL import Image
from skimage import io
from tqdm import tqdm

import chainer
import chainer.functions as F
from chainer import cuda, serializers, Variable

from unet import UNet


class SegmentationModel:

	def __init__(self, model_path, mean, scale=1.0, clahe=True, class_num=5, base_width=32, gpu=0, class_weight=None):

		# Load model
		self._model = UNet(class_num, base_width)
		serializers.load_npz(model_path, self._model)

		if gpu >= 0:
			chainer.cuda.get_device(gpu).use()
			self._model.to_gpu(gpu)
		self._gpu = gpu

		# Add height and width dimensions to mean 
		self._mean = mean[np.newaxis, np.newaxis, :]

		# Scale to resize image
		self._scale = scale

		# Histogram equalization preprocess
		self._clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)) if clahe else None

		# Weight for each class
		self._class_weight = np.ones(shape=[1, 1, class_num]) if (class_weight is None) else np.array([[class_weight]])


	def apply_segmentation(self, pil_image):

		# Down-sample
		if self._scale != 1.0:
			w_orig, h_orig = pil_image.size
			nh, nw = int(h_orig * self._scale), int(w_orig * self._scale)
			pil_image = pil_image.resize((nw, nh), Image.BILINEAR)
		
		image = np.asarray(pil_image, dtype=np.uint8)

		image_in, crop = self._preprocess(image)

		with chainer.using_config('train', False):
			score = self._model.predict(image_in)
		
		score = F.softmax(score)
		score = score.data
		if self._gpu >= 0:
			score = cuda.to_cpu(score)
		score = score[0] # Assuming batch size is 1
		
		top, left, bottom, right = crop
		score = score[:, top:bottom, left:right]
		score = score.transpose(1, 2, 0) # [C, H, W] to [H, W, C]

		# Up-sample
		if self._scale != 1.0:
			score = cv2.resize(score, (w_orig, h_orig), interpolation=cv2.INTER_LINEAR)
		
		# Apply class weight
		score = score * self._class_weight

		# Visualization
		h, w, _ = score.shape
		mask = np.zeros(shape=[h, w, 3], dtype=np.uint8)
		class_idx = np.argmax(score, axis=2)
		mask[class_idx == 0] = np.array([0, 0, 0])      # 0: "background"
		mask[class_idx == 1] = np.array([0, 0, 255])    # 1: "car"
		mask[class_idx == 2] = np.array([255, 0, 0])    # 2: "pedestrian"
		mask[class_idx == 3] = np.array([255, 255, 0])  # 3: "signal"
		mask[class_idx == 4] = np.array([69, 47, 142])  # 4: "lane" (road + parking)
		
		return score, mask


	def _preprocess(self, image):

		h, w, _ = image.shape

		# Histogram equalization
		if self._clahe is not None:
			image_clahe = np.empty(shape=[h, w, 3], dtype=image.dtype)
			for ch in range(3):
				image_clahe[:, :, ch] = self._clahe.apply(image[:, :, ch])
			image = image_clahe
		
		# Padding
		h_padded = int(math.ceil(float(h) / 16.0) * 16)
		w_padded = int(math.ceil(float(w) / 16.0) * 16)

		pad_y1 = (h_padded - h) // 2
		pad_x1 = (w_padded - w) // 2
		pad_y2 = h_padded - h - pad_y1
		pad_x2 = w_padded - w - pad_x1

		image_padded = np.pad(image, ((pad_y1, pad_y2), (pad_x1, pad_x2), (0, 0)), 'symmetric')
		
		# Normalization
		image_in = (image_padded - self._mean) / 255.0
		
		# Reshape and conversion from numpy.ndarray to chainer.Variable
		image_in = image_in.transpose(2, 0, 1)
		image_in = image_in[np.newaxis, :, :, :]

		if self._gpu >= 0:
			image_in = cuda.cupy.asarray(image_in, dtype=cuda.cupy.float32)
		else:
			image_in = np.asarray(image_in, dtype=np.float32)
		image_in = Variable(image_in)

		top, left = pad_y1, pad_x1
		bottom, right = top + h, left + w

		return image_in, (top, left, bottom, right)


if __name__ == '__main__':

	parser = argparse.ArgumentParser(description='Prediction by SS model')
	parser.add_argument('model', help='Path to model weight file')
	parser.add_argument('outdir', help='Output directory')
	parser.add_argument('--base-width', '-bw', type=int, default=32,
						help='Base width of U-Net')
	parser.add_argument('--scale', '-s', type=float, default=0.5)
	parser.add_argument('--noclahe', dest='clahe', action='store_false')
	parser.add_argument('--root', default='../../data/aiedge/seg_test_images')
	parser.add_argument('--mean', default='../../data/aiedge/mean.npy')
	parser.add_argument('--gpu', '-g', type=int, default=0)
	parser.add_argument('--weight', '-w', type=float, nargs=5, default=None)
	args = parser.parse_args()

	mask_dir = os.path.join(args.outdir, 'mask')
	os.makedirs(mask_dir)

	score_dir = os.path.join(args.outdir, 'score')
	os.makedirs(score_dir)

	mean = np.load(args.mean)
	model = SegmentationModel(
		args.model, mean, 
		scale=args.scale, clahe=args.clahe, gpu=args.gpu, base_width=args.base_width, class_weight=args.weight
	)

	image_files = os.listdir(args.root)
	image_files.sort()

	for filename in tqdm(image_files):
		path = os.path.join(args.root, filename)
		pil_image = Image.open(path)

		score, mask = model.apply_segmentation(pil_image)

		basename, _ = os.path.splitext(filename)
		mask_path = os.path.join(mask_dir, '{}.png'.format(basename))
		io.imsave(mask_path, mask)

		score_path = os.path.join(score_dir, '{}.npy'.format(basename))
		np.save(score_path, score)