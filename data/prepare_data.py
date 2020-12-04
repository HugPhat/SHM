import os
import argparse
import random
from shutil import copyfile

from tqdm import tqdm
import cv2 

import sys

sys.path.insert(0, os.getcwd())

import data.gen_trimap as gen_trimap
from data.composite_bg import composite_data_with_voc_bg

def get_args():
    parser = argparse.ArgumentParser(description='gen data folder')
    parser.add_argument('--savedir', type=str, required=True,
                        help="where to save")

    parser.add_argument('--pubdir', type=str, default=None,
                        help="root to public data")
    parser.add_argument('--trimapdir', action='store_false',  default=True,
                        help="if set, ignore trimap")
    parser.add_argument('--prefix', type=str, required=True, default='matte',
                        help="pre/post fix of mask file in folder")
    ### composite image
    parser.add_argument('--vocroot', type=str,  default=None,
                        help="root to pascal voc 2012")
    parser.add_argument('--inimg', type=str,  default=None,
                        help="path to input image folder")
    parser.add_argument('--inalp', type=str,  default=None,
                        help="path to input alpha folder")

    parser.add_argument('--intri', type=str,  default=None,
                        help="path to input trimap folder")
    parser.add_argument('--numbg', type=int, default=100,
                        help="number of random bg")

    parser.add_argument('--trainsplit', type=float, required=True, default=0.8,
                        help="train test split ratio")
    
    args = parser.parse_args()
    print(args)
    return args

def to_sample_name(string, prefix, spliter='_'):
    file_name, file_type = string.split('.')
    file = file_name.split(spliter)
    file.remove(prefix)
    return file[-1] + '.' + file_type

def main(args):
    if args.vocroot != None and (args.inimg == None or args.inalp == None) :
        raise AssertionError('To composite bg, Input/Alpha image is required')
    if args.vocroot != None and args.intri == None and args.trimapdir == False:
        raise AssertionError('trimap folder is required in Trimap Gen')

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
    
    if args.pubdir:
        items = os.listdir(args.pubdir)
        #items_path = [ for each in items]
        print(' ** Gen public data **')
        with tqdm(total=len(items)) as bar:
            for i, each in enumerate(items):
                # get full path of file
                file = os.path.join(args.pubdir, each)
                if args.prefix in each:
                    # to matte folder
                    to_ = 'mask'
                    each = to_sample_name(each, args.prefix, '_')
                    to_dir = os.path.join(mask_image_dir, each)
                    if args.trimapdir:
                        msk = cv2.imread(file)
                        msk = gen_trimap.erode_dilate(
                            msk, struc='ELLIPSE', size=(15, 15))
                        cv2.imwrite(os.path.join(trimap_dir, each), msk)
                else:
                    to_ = 'input'
                    to_dir = os.path.join(input_image_dir, each)

                copyfile(file, to_dir)
                bar.set_description(f'{each} -> {to_}')
                bar.update(1)
    
    if args.vocroot:
        print(' ** Gen Alpha data **')
        composite_data_with_voc_bg(
            voc_bg_root=args.vocroot,
            input_image_folder=args.inimg,
            input_alpha_folder=args.inalp,
            output_folder=args.savedir,
            input_trimap_folder=args.intri,  # '/content/alphamatting_trimap/Trimap1',
            num_random_bg=args.numbg,
        )
    print(' ** save to train/val .txt **')
    items = os.listdir(input_image_dir)
    random.shuffle(items)
    for i,each in enumerate((items)):
        if i < int(len(items) * args.trainsplit):
            train_txt.write(each + '\n')
        else:
            val_txt.write(each + '\n')
    train_txt.close()
    val_txt.close()
    print('Done')

class dummy_arg:
    def __init__(self) -> None:
        self.pubdir = r'E:\ProgrammingSkills\python\DEEP_LEARNING\DATASETS\matting\dataset\training'
        self.savedir = './data/dataset'
        self.prefix = 'matte'
        self.trimapdir = True
        self.trainsplit = 0.8
        
if __name__ == "__main__":
    dummy = get_args() # dummy_arg()
    main(dummy)
