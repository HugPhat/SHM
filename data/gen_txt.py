import os
import argparse
from tqdm import tqdm

import random

def get_args():
    parser = argparse.ArgumentParser(description='gen data folder')
    parser.add_argument('--inpdir', type=str, required=True,
                        help="where to read")
    parser.add_argument('--savedir', type=str, required=True,
                        help="where to save")
    parser.add_argument('--trainsplit', type=float, required=True, default=0.8,
                        help="train test split ratio")
    args = parser.parse_args()
    print(args)
    return args


def main(args):
    items = os.listdir(args.inpdir)
    random.shuffle(items)
    train_txt = open(os.path.join(args.savedir, 'train.txt'), 'w')
    val_txt   = open(os.path.join(args.savedir, 'val.txt'), 'w')

    with tqdm(total=len(items)) as bar:
        for i, each in enumerate(items):
            if i < int(len(items) * args.trainsplit):
                train_txt.write(each + '\n')
                bar.set_description(f'saving to train.txt')
            else:
                val_txt.write(each + '\n')
                bar.set_description(f'saving to val.txt')
            bar.update(1)
    train_txt.close()
    val_txt.close()
    print('Done')


if __name__ == "__main__":
    args = get_args()
    main(args)
