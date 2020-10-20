#!/usr/bin/python

from os import listdir, makedirs
from os.path import exists, join, splitext
from shutil import copy
from random import random


def ensure_directory(dir_path):
    if not exists(dir_path):
        makedirs(dir_path)


def make_dataset_dir(path):
    ensure_directory(path)

    img_dir = join(path, 'images')
    if not exists(img_dir):
        makedirs(img_dir)

    ann_dir = join(path, 'annotations')
    if not exists(ann_dir):
        makedirs(ann_dir)


def get_files_by_ext(dir_path, file_ext):
    return [f for f in listdir(dir_path) if splitext(f)[1] == file_ext]


def main(argv):
    src_dir_path = '/Users/brycegeorge/Git/cs-4800-graphics-project/Data/bowl_data'
    dest_dir_path = '/Users/brycegeorge/Git/bowls2'

    percent_train = 0.8

    img_file_ext = '.jpg'
    ann_file_ext = '.xml'

    img_src_dir = join(src_dir_path, 'images')
    ann_src_dir = join(src_dir_path, 'annotations')
    train_dir = join(dest_dir_path, 'train')
    validation_dir = join(dest_dir_path, 'validation')

    make_dataset_dir(train_dir)
    make_dataset_dir(validation_dir)

    def save_to_train(img_file, ann_file):
        copy(img_file, join(train_dir, 'images'))
        copy(ann_file, join(train_dir, 'annotations'))

    def save_to_validation(img_file, ann_file):
        copy(img_file, join(validation_dir, 'images'))
        copy(ann_file, join(validation_dir, 'annotations'))


    samples = get_files_by_ext(img_src_dir, img_file_ext)

    for sample in samples:
        sample_ann = splitext(sample)[0] + ann_file_ext

        img_file = join(img_src_dir, sample)
        ann_file = join(ann_src_dir, sample_ann)

        if not exists(ann_file):
            continue

        if random() < percent_train:
            save_to_train(img_file, ann_file)
        else:
            save_to_validation(img_file, ann_file)




