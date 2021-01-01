import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.backends import cudnn

from torch.utils.tensorboard import SummaryWriter

import numpy as np

from models.shm import create_shm

from utils.data import make_image_to_infer, tensor_2_npImage

cudnn.benchmark = True


class SHMInf(object):
    def __init__(self, config):
        super().__init__()
        self.config = config
        #print(
        #    "Creating SHM architecture and loading pretrained weights...")

        self.model = create_shm(backbone=config.backbone)

        self.cuda = torch.cuda.is_available() & self.config.cuda

        if self.cuda:
            self.device = torch.device("cuda")
            print("Operation will be on *****GPU-CUDA***** ")
        else:
            self.device = torch.device("cpu")
            print("Operation will be on *****CPU***** ")
        self.load_checkpoint()
        self.model.to(self.device)
        self.model.eval()

    def load_checkpoint(self):
        assert self.config.model in ['tnet',
                                     'mnet', 'end2end']
        try:
            if self.config.model == 'tnet':
                self.model = self.model.tnet
                if not self.config.pretrained_tnet is None:
                  try:
                    self.model.load_state_dict(
                          torch.load(self.config.pretrained_tnet)['state_dict'] )
                  except:
                    self.model.load_state_dict(
                          torch.load(self.config.pretrained_tnet))
                  print(f'Loaded pretrained {self.config.model}')
            elif self.config.model == 'mnet':
                self.model = self.model.mnet     
                if not self.config.pretrained_mnet is None:
                  try:
                    self.model.load_state_dict(
                          torch.load(self.config.pretrained_mnet)['state_dict'] )
                  except:
                    self.model.load_state_dict(
                          torch.load(self.config.pretrained_mnet))
                 
                  print(f'Loaded pretrained {self.config.model}')
            elif self.config.model == 'end2end':
                if not self.config.pretrained_end2end is None:
                    self.model.load_state_dict(
                        torch.load(self.config.pretrained_end2end)['state_dict'])
                    print(f'Loaded pretrained {self.config.model}')
                elif not self.config.pretrained_tnet is None:
                    self.model.tnet.load_state_dict(
                        torch.load(self.config.pretrained_tnet)['state_dict'] )
                    print(f'Loaded pretrained tnet of {self.config.model}')
                elif not self.config.pretrained_mnet is None:
                    self.model.tnet.load_state_dict(
                        torch.load(self.config.pretrained_mnet)['state_dict'])
                    print(f'Loaded pretrained mnet of {self.config.model}')

            
        except OSError as e:
            print("No checkpoint exists. Skipping...")
            

    def trimap_to_image(self, trimap):
        n, c, h, w = trimap.size()
        if c == 3:
            trimap = torch.argmax(trimap, dim=1, keepdim=False)
        return trimap.float().div_(2.0).view(n, 1, h, w)

    def alpha_to_image(self, alpha):
        return alpha.clamp_(0, 1)

    def predict(self, image: np.ndarray, trimap: np.ndarray=None) -> list:
        """
        Predict alpha or trimap
            returns type:
                if mnet: [None, alpha]
                if tnet : [trimap, None]
                if end2end : [trimap, alpha] 
        Args:
            image (np.ndarray)  : RGB image
            trimap (np.ndarray) : trimap 
        Returns:
            list(trimap, alpha)
        """
        self.model.eval()
        Timage = make_image_to_infer(self.config.model, image, trimap)
        #print(Timage.size())
        with torch.no_grad():
            Timage = Timage.to(self.device)
            output = self.model(Timage)
        if self.config.model == 'tnet':
            #trimap
            output = self.trimap_to_image(
                output.cpu())
            output = tensor_2_npImage(output, nrow=1, padding=0)
            return [output, None]
        elif self.config.model == 'mnet':
            #alpha
            output = self.alpha_to_image(output.cpu())
            output = (output.squeeze(0)*255.).squeeze(0).numpy() #tensor_2_npImage(output, nrow=1, padding=0)
            #output = tensor_2_npImage(self.alpha_to_image(output), nrow=1, padding=0)
            return [None, output]
        else:
            #matte, alpha
            matte, alpha = output[0], output[1]
            matte = tensor_2_npImage(matte, nrow=1, padding=0)
            alpha = tensor_2_npImage(alpha, nrow=1, padding=0)
            return [matte, alpha]
        
