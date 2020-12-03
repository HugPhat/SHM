import argparse
from datasets.test_data import * 


def get_args():
    parser = argparse.ArgumentParser(description='gen data folder')
    parser.add_argument('--inpdir', type=str, required=True,
                        help="where to read")
    parser.add_argument('--type', type=str, required=True, default='pretrain_tnet',
                        help="pretrain_tnet, pretrain_mnet")

    args = parser.parse_args()
    print(args)
    return args

if __name__ == "__main__":
    args = get_args()

    data_loader = TestDatasetDataLoader(
        args.inpdir, args.type, 4).train_loader
    image, trimap, alpha = next(iter(data_loader))
    print(image.size(), trimap.size(), alpha.size())
    print(image)
    print(trimap)
    print(alpha)
