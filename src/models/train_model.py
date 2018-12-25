#!/usr/bin/env python

from __future__ import print_function

import argparse
import numpy as np

import chainer
import chainer.functions as F
import chainer.links as L
from chainer import training
from chainer.training import extensions

from unet import UNet
from dataset import LabeledImageDataset

from tensorboardX import SummaryWriter
from tboard_logger import TensorboardLogger
from iou_evaluator import IouEvaluator

import os


def train_model():
	parser = argparse.ArgumentParser()
	parser.add_argument('data_type', choices=['cityscapes', 'aiedge'])
	parser.add_argument('--arch', '-a', choices=['unet'], default='unet')
	parser.add_argument('--scale', '-s', type=float, default=0.5,
						help='Scale factor to resize images')
	parser.add_argument('--tcrop', '-t', type=int, nargs=2, default=[1024, 512],
						help='Crop size for train images [w, h]')
	parser.add_argument('--vcrop', '-v', type=int, nargs=2, default=[1024, 512],
						help='Crop size for train images [w, h]')
	parser.add_argument('--batchsize', '-b', type=int, default=5,
						help='Number of images in each mini-batch')
	parser.add_argument('--test-batchsize', '-B', type=int, default=1,
						help='Number of images in each test mini-batch')
	parser.add_argument('--epoch', '-e', type=int, default=100,
						help='Number of sweeps over the dataset to train')
	parser.add_argument('--frequency', '-f', type=int, default=1,
						help='Frequency of taking a snapshot')
	parser.add_argument('--gpu', '-g', type=int, default=0,
						help='GPU ID (negative value indicates CPU)')
	parser.add_argument('--out', '-o', default='log_00',
						help='Directory to output the result under "models" directory')
	parser.add_argument('--weight', '-w', default=None,
						help='Path to pretrained model to initialize the weight')
	parser.add_argument('--resume', '-r', default='',
						help='Resume the training from snapshot')
	parser.add_argument('--noplot', dest='plot', action='store_false',
						help='Disable PlotReport extension')
	args = parser.parse_args()

	if args.data_type == 'cityscapes':
		data_root = '../../data/cityscapes'
		color_distort = True
	if args.data_type == 'aiedge':
		data_root = '../../data/cityscapes'
		#color_distort = False
		pass
	
	print('Data type: {}'.format(args.data_type))
	print('# Image scale: {}'.format(args.scale))
	print('# Train crop-size: {}'.format(args.tcrop))
	print('# Test crop-size: {}'.format(args.vcrop))
	print('# Minibatch-size: {}'.format(args.batchsize))
	print('# Epoch: {}'.format(args.epoch))
	print('# GPU: {}'.format(args.gpu))
	print('')
	
	this_dir = os.path.dirname(os.path.abspath(__file__))
	models_dir = os.path.normpath(os.path.join(this_dir, "../../models"))
	log_dir = os.path.join(models_dir, args.out)
	writer = SummaryWriter(log_dir=log_dir)
	
	# Set up a neural network to train
	# Classifier reports softmax cross entropy loss and accuracy at every
	# iteration, which will be used by the PrintReport extension below.
	if args.arch == 'unet':
		model = UNet(class_num=5)
	if args.weight is not None:
		chainer.serializers.load_npz(args.weight, model)
	if args.gpu >= 0:
		# Make a specified GPU current
		chainer.cuda.get_device_from_id(args.gpu).use()
		model.to_gpu()  # Copy the model to the GPU

	# Setup an optimizer
	optimizer = chainer.optimizers.Adam()
	optimizer.setup(model)
	
	# Load mean image
	mean = np.load(os.path.join(data_root, "mean.npy"))
	
	# Load the MNIST dataset
	train = LabeledImageDataset(args.data_type, os.path.join(data_root, "train.txt"), data_root, args.tcrop, scale=args.scale,
								mean=mean, random_crop=True, hflip=True, color_distort=color_distort)
	
	test = LabeledImageDataset (args.data_type, os.path.join(data_root, "val.txt"), data_root, args.vcrop, scale=args.scale,
								mean=mean, random_crop=False, hflip=False, color_distort=False)

	train_iter = chainer.iterators.SerialIterator(train, args.batchsize)
	test_iter = chainer.iterators.SerialIterator(test, args.test_batchsize, repeat=False, shuffle=False)

	# Set up a trainer
	updater = training.StandardUpdater(
		train_iter, optimizer, device=args.gpu)
	trainer = training.Trainer(updater, (args.epoch, 'epoch'), out=log_dir)

	# Evaluate the model with the test dataset for each epoch
	#trainer.extend(extensions.Evaluator(test_iter, model, device=args.gpu))
	label_names = ['car', 'pedestrian', 'signal', 'lane']
	trainer.extend(IouEvaluator(test_iter, model, device=args.gpu, label_names=label_names))

	# Dump a computational graph from 'loss' variable at the first iteration
	# The "main" refers to the target link of the "main" optimizer.
	trainer.extend(extensions.dump_graph('main/loss'))

	# Take a snapshot for each specified epoch
	frequency = args.epoch if args.frequency == -1 else max(1, args.frequency)
	trainer.extend(extensions.snapshot(
		filename='snapshot_epoch_{.updater.epoch}'), trigger=(frequency, 'epoch'))
	
	# Save trained model for each specific epoch
	trainer.extend(extensions.snapshot_object(
		model, filename='model_epoch_{.updater.epoch}'), trigger=(frequency, 'epoch'))

	# Write a log of evaluation statistics for each epoch
	trainer.extend(extensions.LogReport())

	# Save two plot images to the result dir
	if args.plot and extensions.PlotReport.available():
		trainer.extend(
			extensions.PlotReport(
				['main/loss', 'validation/main/loss'],
				'epoch', file_name='loss.png'))
		trainer.extend(
			extensions.PlotReport(
				['main/accuracy', 'validation/main/accuracy'],
				'epoch', file_name='accuracy.png'))
		trainer.extend(
			extensions.PlotReport(
				['iou'],
				'epoch', file_name='iou.png'))

	# Print selected entries of the log to stdout
	# Here "main" refers to the target link of the "main" optimizer again, and
	# "validation" refers to the default name of the Evaluator extension.
	# Entries other than 'epoch' are reported by the Classifier link, called by
	# either the updater or the evaluator.
	trainer.extend(extensions.PrintReport(
		['epoch', 'iou', 'main/loss', 'validation/main/loss',
		 'main/accuracy', 'validation/main/accuracy', 'elapsed_time']))

	# Print a progress bar to stdout
	trainer.extend(extensions.ProgressBar())
	
	# Write training log to TensorBoard log file
	trainer.extend(TensorboardLogger(writer, [
		'main/loss', 'main/accuracy', 
		'validation/main/loss', 'validation/main/accuracy'
		], x='iteration'))

	entries = entries = ['iou']
	for label_name in label_names:
		entries.append('iou/{:s}'.format(label_name))
	trainer.extend(TensorboardLogger(writer, entries, x='epoch'))
	
	if args.resume:
		# Resume from a snapshot
		chainer.serializers.load_npz(args.resume, trainer)

	# Run the training
	trainer.run()


if __name__ == '__main__':
	train_model()
