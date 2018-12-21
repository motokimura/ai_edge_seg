#!/usr/bin/env python

import argparse
import os

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', '-d', default='../../data/cityscapes',
                        help='Root data directory containing `gtFine_trainvaltest` and `leftImg8bit_trainvaltest`')
    parser.add_argument('--image_dir', '-i', default='leftImg8bit_trainvaltest/leftImg8bit')
    parser.add_argument('--label_dir', '-l', default='gtFine_trainvaltest/gtFine')
    return parser.parse_args()

def dump_data_list(args, split):
    image_root = os.path.join(args.data_root, args.image_dir)
    label_root = os.path.join(args.data_root, args.label_dir)

    image_dir = os.path.join(image_root, split)
    label_dir = os.path.join(label_root, split)
    out_path = os.path.join(args.data_root, '{}.txt'.format(split))

    with open(out_path, 'w') as f:
        
        city_names = os.listdir(image_dir)
        for city in city_names:
            image_city_dir = os.path.join(image_dir, city)
            label_city_dir = os.path.join(label_dir, city)
            
            image_filenames = os.listdir(image_city_dir)
            for i, image_file in enumerate(image_filenames):
                basename = image_file[:-16] # remove '_leftImg8bit.png'
                label_file = '{}_gtFine_labelIds.png'.format(basename)

                # Check if the label image exists or not
                label_path = os.path.join(label_city_dir, label_file)
                assert os.path.exists(label_path)

                # Dump
                image_rel_path = os.path.join(args.image_dir, split, city, image_file)
                label_rel_path = os.path.join(args.label_dir, split, city, label_file)
                f.write('{},{}\n'.format(image_rel_path, label_rel_path))

def main():
    args = parse_args()

    for split in ['train', 'val', 'test']:
        dump_data_list(args, split)

if __name__ == '__main__':
    main()
