#!/usr/bin/env python

import os
import numpy as np
import random
import cv2
import math

try:
	from PIL import Image
	available = True
except ImportError as e:
	available = False
	_import_error = e
import six

from chainer.dataset import dataset_mixin

from transforms import random_color_distort


def _check_pillow_availability():
	if not available:
		raise ImportError('PIL cannot be loaded. Install Pillow!\n'
						  'The actual import error is as follows:\n' +
						  str(_import_error))


def _read_image_as_array(path, dtype, scale, resample=Image.NEAREST):
	f = Image.open(path)

	if scale != 1.0:
		w, h = f.size
		nh, nw = int(h * scale), int(w * scale)
		f = f.resize((nw, nh), resample)

	try:
		image = np.asarray(f, dtype=dtype)
	finally:
		# Only pillow >= 3.0 has 'close' method
		if hasattr(f, 'close'):
			f.close()
	return image


class LabeledImageDataset(dataset_mixin.DatasetMixin):
	def __init__(self, data_type, dataset, root, crop_wh, scale=1,
				 dtype=np.float32, label_dtype=np.int32, mean=None, clahe=False,
				 random_crop=False, hflip=False, color_distort=False, pad=0):
		assert data_type in ['cityscapes', 'aiedge']
		_check_pillow_availability()
		if isinstance(dataset, six.string_types):
			dataset_path = dataset
			with open(dataset_path) as f:
				pairs = []
				for i, line in enumerate(f):
					line = line.rstrip('\n')
					image_filename, label_filename = line.split(',')
					pairs.append((image_filename, label_filename))
		self._pairs = pairs
		self._root = root
		self._dtype = dtype
		self._label_dtype = label_dtype
		self._normalize = False if (mean is None) else True
		if self._normalize:
			self._mean = mean[np.newaxis, np.newaxis, :]
		self._crop_w, self._crop_h = crop_wh
		self._scale = scale
		self._random_crop = random_crop
		self._hflip = hflip
		self._color_distort = color_distort
		self._pad = pad
		self._clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)) if clahe else None

		self._get_label = self._get_cityscapes_label if data_type == 'cityscapes' else self._get_aiedge_label

	def __len__(self):
		return len(self._pairs)
	
	def _get_cityscapes_label(self, label_image):
		h, w = label_image.shape[0], label_image.shape[1]
		label = np.zeros(shape=[h, w], dtype=self._label_dtype)  # 0: "background"
		label[label_image == 26] = 1                             # 1: "car"
		label[label_image == 24] = 2                             # 2: "pedestrian"
		label[label_image == 19] = 3                             # 3: "signal"
		label[(label_image == 7) | (label_image == 9)] = 4       # 4: "lane" (road + parking)
		return label
	
	def _get_aiedge_label(self, label_image):
		h, w = label_image.shape[0], label_image.shape[1]
		label = np.zeros(shape=[h, w], dtype=self._label_dtype)  # 0: "background"
		label[(label_image==[0, 0, 255]).sum(axis=2) == 3] = 1	 # 1: "car"
		label[(label_image==[255, 0, 0]).sum(axis=2) == 3] = 2	 # 2: "pedestrian"
		label[(label_image==[255, 255, 0]).sum(axis=2) == 3] = 3 # 3: "signal"
		label[(label_image==[69, 47, 142]).sum(axis=2) == 3] = 4 # 4: "lane" (road + parking)
		return label

	def get_example(self, i):
		image_filename, label_filename = self._pairs[i]
		
		image_path = os.path.join(self._root, image_filename)
		image = _read_image_as_array(image_path, np.uint8, self._scale, Image.BILINEAR)
		
		label_path = os.path.join(self._root, label_filename)
		label_image = _read_image_as_array(label_path, self._label_dtype, self._scale, Image.NEAREST)
		label = self._get_label(label_image)

		h, w, _ = image.shape

		# Histogram equalization
		if self._clahe is not None:
			image_clahe = np.empty(shape=[h, w, 3], dtype=image.dtype)
			for ch in range(3):
				image_clahe[:, :, ch] = self._clahe.apply(image[:, :, ch])
			image = image_clahe

		if h < self._crop_h:
			# Padding
			pad_top = (self._crop_h - h) // 2
			pad_bottom = self._crop_h - h - pad_top
			image = np.pad(image, [(pad_top, pad_bottom), (0, 0), (0, 0)], 'symmetric')
			label = np.pad(label, [(pad_top, pad_bottom), (0, 0)], 'constant', constant_values=255)
			h = self._crop_h
		
		if w < self._crop_w:
			# Padding
			pad_left = (self._crop_w - w) // 2
			pad_right = self._crop_w - w - pad_left
			image = np.pad(image, [(0, 0), (pad_left, pad_right), (0, 0)], 'symmetric')
			label = np.pad(label, [(0, 0), (pad_left, pad_right)], 'constant', constant_values=255)
			w = self._crop_w
		
		if self._pad > 0:
			pad = int(self._pad * self._scale)
			image = np.pad(image, [(pad, pad), (pad, pad), (0, 0)], 'symmetric')
			label = np.pad(label, [(pad, pad), (pad, pad)], 'constant', constant_values=255)
			h = h + 2 * pad
			w = w + 2 * pad

		if self._random_crop:
			# Random crop
			top  = random.randint(0, h - self._crop_h)
			left = random.randint(0, w - self._crop_w)
		else:
			# Crop center
			top = (h - self._crop_h) // 2
			left = (w - self._crop_w) // 2
		
		bottom = top + self._crop_h
		right = left + self._crop_w

		image = image[top:bottom, left:right]
		label = label[top:bottom, left:right]

		if self._hflip:
			# Horizontal flip
			if random.randint(0, 1):
				image = image[:, ::-1, :]
				label = label[:, ::-1]
		
		if self._color_distort:
			image = random_color_distort(image)
		
		# Preprocess
		image = np.asarray(image, dtype=np.float64)
		if self._normalize:
			image = (image - self._mean) / 255.0
		
		# Type casting
		image = image.astype(self._dtype) 
		label = self._label_dtype(label)
			
		return image.transpose(2, 0, 1), label