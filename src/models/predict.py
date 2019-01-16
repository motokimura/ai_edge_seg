#!/usr/bin/env python
import argparse
import os

import numpy as np
import cv2
import math
from PIL import Image, ImageOps
from skimage import io
from tqdm import tqdm

import chainer
import chainer.functions as F
from chainer import cuda, serializers, Variable

from unet import UNet
from dilated_unet import DilatedUNet


class SegmentationModel:

	def __init__(self, model_path, mean, 
		arch='unet', scale=1.0, clahe=True, class_num=5, base_width=32, bn=True, gpu=0):

		assert arch in ['unet', 'dilated']

		self._arch = arch
		self._model_path = model_path
		self._base_width = base_width
		self._bn = bn
		self._class_num = class_num
		self._gpu = gpu

		# Load model
		self.load_weight()

		# Add height and width dimensions to mean 
		self._mean = mean[np.newaxis, np.newaxis, :]

		# Scale to resize image
		self._scale = scale

		# Histogram equalization preprocess
		self._clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)) if clahe else None
	
	def load_weight(self):
		if self._arch == 'unet':
			self._model = UNet(self._class_num, self._base_width)
		if self._arch == 'dilated':
			self._model = DilatedUNet(self._class_num, self._base_width, self._bn)
		serializers.load_npz(self._model_path, self._model)

		if self._gpu >= 0:
			chainer.cuda.get_device(self._gpu).use()
			self._model.to_gpu(self._gpu)

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

		# Visualization
		mask = self.make_mask(score)
		
		return score, mask

	def make_mask(self, score):
		h, w, _ = score.shape
		mask = np.zeros(shape=[h, w, 3], dtype=np.uint8)
		class_idx = np.argmax(score, axis=2)
		mask[class_idx == 0] = np.array([0, 0, 0])      # 0: "background"
		mask[class_idx == 1] = np.array([0, 0, 255])    # 1: "car"
		mask[class_idx == 2] = np.array([255, 0, 0])    # 2: "pedestrian"
		mask[class_idx == 3] = np.array([255, 255, 0])  # 3: "signal"
		mask[class_idx == 4] = np.array([69, 47, 142])  # 4: "lane" (road + parking)

		return mask

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
	parser.add_argument('outdir', help='Output directory')
	parser.add_argument('--models', '-m', nargs='+', help='Paths to models weight file')
	parser.add_argument('--ens-weights', '-w', type=float, nargs='+', default=None)
	parser.add_argument('--cat-factor', '-c', type=float, nargs=5, default=None)
	parser.add_argument('--flip', action='store_true')
	parser.add_argument('--arch', '-a', choices=['unet', 'dilated'], default='dilated')
	parser.add_argument('--bn', action='store_true', help='Use batch-normalization')
	parser.add_argument('--base-width', '-bw', type=int, default=44,
						help='Base width of U-Net')
	parser.add_argument('--scale', '-s', type=float, default=1.0)
	parser.add_argument('--noclahe', dest='clahe', action='store_false')
	parser.add_argument('--root', default='../../data/aiedge/seg_test_images')
	parser.add_argument('--mean', default='../../data/aiedge/mean.npy')
	parser.add_argument('--gpu', '-g', type=int, default=0)
	args = parser.parse_args()

	N = len(args.models)
	if args.ens_weights is None:
		ens_weights = np.ones(shape=[N,])
	else:
		assert len(args.ens_weights) == N
		ens_weights = np.array(args.ens_weights)
	ens_weights /= N

	cat_factor = np.ones(shape=[1, 1, 5]) if (args.cat_factor is None) else np.array([[args.cat_factor]])
	cat_factor /= 5

	mask_dir = os.path.join(args.outdir, 'mask')
	os.makedirs(mask_dir)

	score_dir = os.path.join(args.outdir, 'score')
	os.makedirs(score_dir)

	mean = np.load(args.mean)

	image_files = os.listdir(args.root)
	image_files.sort()

	for filename in tqdm(image_files):
		path = os.path.join(args.root, filename)
		pil_image = Image.open(path)

		w, h = pil_image.size
		score_ensemble = np.zeros(shape=[h, w, 5], dtype=np.float32)

		for model_path, ens_weight in zip(args.models, ens_weights):
			model = SegmentationModel(
				model_path, mean, arch=args.arch, scale=args.scale, clahe=args.clahe, class_num=5,
				base_width=args.base_width, bn=args.bn, gpu=args.gpu
			)
			model.load_weight()
			
			score, _ = model.apply_segmentation(pil_image)
			if args.flip:
				image_flip = ImageOps.mirror(pil_image)
				score_flip = model.apply_segmentation(pil_image)
				score = (score + score_flip[:, ::-1, :]) / 2

			score_ensemble += score * ens_weight

		score_ensemble *= cat_factor
		mask_ensemble = model.make_mask(score_ensemble)

		basename, _ = os.path.splitext(filename)
		mask_path = os.path.join(mask_dir, '{}.png'.format(basename))
		io.imsave(mask_path, mask_ensemble)

		score_path = os.path.join(score_dir, '{}.npy'.format(basename))
		np.save(score_path, score_ensemble)