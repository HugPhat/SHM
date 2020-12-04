import os
import argparse
import random

from tqdm import tqdm

import sys

sys.path.insert(0, os.getcwd())

from data.composite_bg import composite_data_with_voc_bg

def get_args():
    parser = argparse.ArgumentParser(description='gen data folder')
    parser.add_argument('--savedir', type=str, required=True,
                        help="where to save")
    ### composite image
    parser.add_argument('--vocroot', type=str,  required=True,
                        help="root to pascal voc 2012")
    parser.add_argument('--inimg', type=str,  required=True,
                        help="path to input image folder")
    parser.add_argument('--inalp', type=str,  required=True,
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


def main(args):
    input_image_dir = os.path.join(args.savedir, 'input')

    train_txt = open(os.path.join(args.savedir, 'train.txt'), 'w')
    val_txt = open(os.path.join(args.savedir, 'val.txt'), 'w')
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
    for i, each in enumerate((items)):
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
    dummy = get_args()  # dummy_arg()
    main(dummy)
