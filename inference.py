import argparse
import warnings
warnings.filterwarnings('ignore')
from utils.config import *
import cv2
from easydict import EasyDict
from agents.shm_infer import SHMInf
import matplotlib.pyplot as plt
import yaml 

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

    with open(args.config, 'r') as config_file:
      config_dict = yaml.load(config_file)
      config = EasyDict(config_dict)
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
