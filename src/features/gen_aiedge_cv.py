#!/usr/bin/env python

import argparse
import os
import random
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', '-d', default='../../data/aiedge',
                        help='Root data directory containing `seg_train_images` and `seg_train_annotations`')
    parser.add_argument('--image_dir', '-i', default='seg_train_images')
    parser.add_argument('--label_dir', '-l', default='seg_train_annotations')
    parser.add_argument('--split', '-s', type=int, default=5,
                        help='Split ratio for train/val')
    parser.add_argument('--seed', type=int, default=0,
                        help='Random seed')
    return parser.parse_args()

def dump_data_list(split, data_root, image_dir, label_dir, filenames):
    out_path = os.path.join(data_root, '{}.txt'.format(split))

    with open(out_path, 'w') as f:
        for image_file in filenames:
            basename = image_file[:-4] # remove '.jpg'
            label_file = '{}.png'.format(basename)

            image_rel_path = os.path.join(image_dir, image_file)
            label_rel_path = os.path.join(label_dir, label_file)
            f.write('{},{}\n'.format(image_rel_path, label_rel_path))

def main():
    args = parse_args()
    
    image_root = os.path.join(args.data_root, args.image_dir)
    image_filenames = os.listdir(image_root)
    image_filenames.sort()
    random.seed(args.seed)
    random.shuffle(image_filenames)

    splits = np.array_split(image_filenames, args.split)

    for i in range(args.split):
        val_filenames = list(splits[i])
        
        train_filenames = []
        for j in range(args.split):
            if j == i:
                continue
            train_filenames.extend(list(splits[j]))
        
        # Gen train_{i}.txt and train_{i}.txt for i in range(args.split)
        dump_data_list('train_{}'.format(i), args.data_root, args.image_dir, args.label_dir, train_filenames)
        dump_data_list('val_{}'.format(i), args.data_root, args.image_dir, args.label_dir, val_filenames)
    
    # Gen train.txt used for mean cumputation
    dump_data_list('train', args.data_root, args.image_dir, args.label_dir, image_filenames)

if __name__ == '__main__':
    main()