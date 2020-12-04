import argparse
import warnings
warnings.filterwarnings('ignore')
from utils.config import *
import cv2
from agents import SHMInf

import matplotlib.pyplot as plt

def main():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument(
        '--config',
        default=None,
        help='The Path of configuration file in yaml format')
    arg_parser.add_argument(
        '--img',
        default=None,
        help='The Path of image')
    arg_parser.add_argument(
        '--trimap',
        default=None,
        help='The Path of trimap')
    args = arg_parser.parse_args()
    config = process_config(args.config)
    
    if config.model == 'mnet' and args.trimap == None:
        raise AssertionError('trimap is require in mnet model')
    
    agent = SHMInf(config)
    img = cv2.imread(args.img)
    trimap = None if args.trimap is None else cv2.imread(args.trimap)
    Img, Tri = agent.predict(img, trimap)
    if Tri and Img:
        f, axarr = plt.subplots(1, 2)
        axarr[0, 0].imshow(Img)
        axarr[0, 1].imshow(Tri)
    elif Img:
        plt.imshow(Img)
    else:
        plt.imshow(Tri)
    plt.show()


if __name__ == '__main__':
    main()
