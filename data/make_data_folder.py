import os 
import argparse
from tqdm import tqdm
from shutil import copyfile


def get_args():
    parser = argparse.ArgumentParser(description='gen data folder')
    parser.add_argument('--inpdir', type=str, required=True,
                        help="where to read")
    parser.add_argument('--savedir', type=str, required=True,
                        help="where to save")
    parser.add_argument('--prefix', type=str, required=True, default='matte',
                        help="list of images id")
    args = parser.parse_args()
    print(args)
    return args

def main(args):
    items = os.listdir(args.inpdir)
    input_image_dir = os.path.join(args.savedir, 'input')
    mask_image_dir  = os.path.join(args.savedir, 'mask')
    if not os.path.exists(input_image_dir):
        os.mkdir(input_image_dir)
    if not os.path.exists(mask_image_dir):
        os.mkdir(mask_image_dir)

    #items_path = [ for each in items]
    with tqdm(total=len(items)) as bar:
        for each in items:
            # get full path of file
            file = os.path.join(args.inpdir, each)
            if args.prefix in each:
                # to matte folder
                to_ = 'mask'
                to_dir = os.path.join(mask_image_dir, each)
            else:
                to_ ='input'
                to_dir = os.path.join(input_image_dir, each)
            copyfile(file, to_dir)
            bar.set_description(f'{each} -> {to_}')
            bar.update(1)

class dummy_arg:
    def __init__(self) -> None:
        self.inpdir = r'E:\ProgrammingSkills\python\DEEP_LEARNING\DATASETS\matting\dataset\training'
        self.savedir = './data'
        self.prefix = 'matte'

if __name__ == "__main__":
    dummy = dummy_arg()
    main(dummy)