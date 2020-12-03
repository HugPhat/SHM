import os
import argparse
from shutil import copyfile

from tqdm import tqdm
import cv2 

import sys

sys.path.insert(0, os.getcwd())

import data.gen_trimap as gen_trimap

def get_args():
    parser = argparse.ArgumentParser(description='gen data folder')
    parser.add_argument('--inpdir', type=str, required=True,
                        help="where to read")
    parser.add_argument('--savedir', type=str, required=True,
                        help="where to save")
    parser.add_argument('--trimapdir', action='store_false',  default=True,
                        help="if set, ignore trimap")
    parser.add_argument('--trainsplit', type=float, required=True, default=0.8,
                        help="train test split ratio")

    parser.add_argument('--prefix', type=str, required=True, default='matte',
                        help="pre/post fix of mask file in folder")

    args = parser.parse_args()
    print(args)
    return args

def to_sample_name(string, prefix, spliter='_'):
    file_name, file_type = string.split('.')
    file = file_name.split(spliter)
    file.remove(prefix)
    return file[-1] + '.' + file_type

def main(args):
    items = os.listdir(args.inpdir)
    input_image_dir = os.path.join(args.savedir, 'input')
    mask_image_dir = os.path.join(args.savedir, 'mask')
    if not os.path.exists(input_image_dir):
        os.makedirs(input_image_dir)
    if not os.path.exists(mask_image_dir):
        os.makedirs(mask_image_dir)
    if args.trimapdir:
        trimap_dir = os.path.join(args.savedir, 'trimap')
        if not os.path.exists(trimap_dir):
            os.mkdir(trimap_dir)
    train_txt = open(os.path.join(args.savedir,'train.txt'), 'w')
    val_txt = open(os.path.join(args.savedir,'val.txt'), 'w')
    #items_path = [ for each in items]
    with tqdm(total=len(items)) as bar:
        for i, each in enumerate(items):
            # get full path of file
            file = os.path.join(args.inpdir, each)
            if args.prefix in each:
                # to matte folder
                to_ = 'mask'
                each = to_sample_name(each, args.prefix, '_')
                to_dir = os.path.join(mask_image_dir, each)
                if args.trimapdir:
                    msk = cv2.imread(file)
                    msk = gen_trimap.erode_dilate(
                        msk, struc='ELLIPSE', size=(10, 10))
                    cv2.imwrite(os.path.join(trimap_dir, each), msk)
                if i < int(len(items) * args.trainsplit):
                    train_txt.write(each + '\n')
                else:
                    val_txt.write(each + '\n')
            else:
                to_ = 'input'
                to_dir = os.path.join(input_image_dir, each)
                if i < int(len(items) * args.trainsplit):
                    train_txt.write(each + '\n')
                else:
                    val_txt.write(each + '\n')

            copyfile(file, to_dir)
            bar.set_description(f'{each} -> {to_}')
            bar.update(1)
    train_txt.close()
    val_txt.close()
    print('Done')

class dummy_arg:
    def __init__(self) -> None:
        self.inpdir = r'E:\ProgrammingSkills\python\DEEP_LEARNING\DATASETS\matting\dataset\training'
        self.savedir = './data/dataset'
        self.prefix = 'matte'
        self.trimapdir = True
        self.trainsplit = 0.8
        
if __name__ == "__main__":
    dummy = get_args() # dummy_arg()
    main(dummy)
