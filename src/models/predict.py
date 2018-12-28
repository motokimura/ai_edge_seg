#!/usr/bin/env python

import numpy as np
import cv2
import math
from PIL import Image

import chainer
import chainer.functions as F
from chainer import cuda, serializers, Variable

from unet import UNet


class SegmentationModel:

	def __init__(self, model_path, mean, scale=1.0, clahe=True, class_num=5, base_width=32, gpu=0):

		# Load model
		self._model = UNet(class_num, base_width)
		serializers.load_npz(model_path, self._model)

		chainer.cuda.get_device(gpu).use()
		self._model.to_gpu(gpu)

		# Add height and width dimensions to mean 
		self._mean = mean[np.newaxis, np.newaxis, :]

		# Scale to resize image
		self._scale = scale

		# Histogram equalization preprocess
		self._clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)) if clahe else None


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
		score = cuda.to_cpu(score.data)[0]
		
		top, left, bottom, right = crop
		score = score[:, top:bottom, left:right]
		score = score.transpose(1, 2, 0) # [C, H, W] to [H, W, C]

		# Up-sample
		if self._scale != 1.0:
			score = cv2.resize(score, (w_orig, h_orig), interpolation=cv2.INTER_LINEAR)

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
		image_in = Variable(cuda.cupy.asarray(image_in, dtype=cuda.cupy.float32))

		top, left = pad_y1, pad_x1
		bottom, right = top + h, left + w

		return image_in, (top, left, bottom, right)
