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
    h,w = img.shape[:2]
    h = h-1 if h%2 != 0 else h
    w = w-1 if w%2 != 0 else w
    img = cv2.resize(img, (w,h))

    trimap = None if args.trimap is None else cv2.imread(args.trimap, 0)
    if type(trimap) != None:
      trimap = cv2.resize(trimap, (w,h), interpolation= cv2.INTER_NEAREST)
    Tri, Img = agent.predict(img, trimap)
    plt.rcParams["figure.figsize"] = (20,8)
    if not Tri is None and not Img is None :
        f, axarr = plt.subplots(1, 2)
        axarr[0].imshow(Img)
        axarr[1].imshow(Tri)
    elif type(Img) != None:
        cv2.imwrite('img.jpg', (Img*255.).astype('uint8') )
        plt.imshow(Img)
    else:
        plt.imshow(Tri)
    plt.show()


if __name__ == '__main__':
    main()
