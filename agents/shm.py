import os
import logging
import torch
import torch.nn as nn
import torch.optim as optim
from torch.backends import cudnn
from torchvision.utils import make_grid, save_image
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from models.shm import create_shm
from models.loss import PredictionL1Loss, ClassificationLoss
from datasets.test_data import TestDatasetDataLoader as Data_loader
from utils.metrics import AverageMeter, accuracy
from utils.misc import print_cuda_statistics
from utils.data import make_sample

cudnn.benchmark = True


class SHMAgent(object):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.logger = logging.getLogger("SHMAgent")
        self.logger.info("Creating SHM architecture and loading pretrained weights...")

        self.model = create_shm(backbone=config.backbone)
        self.data_loader = Data_loader(self.config.data_root, self.config.mode, self.config.batch_size)
        if self.config.eval:
            self.eval_loader = Data_loader(
                self.config.data_root, self.config.mode, self.config.batch_size, True)

        self.current_epoch = 0
        self.cuda = torch.cuda.is_available() & self.config.cuda
        if self.cuda:
            self.device = torch.device("cuda")
            self.logger.info("Operation will be on *****GPU-CUDA***** ")
            print_cuda_statistics()
        else:
            self.device = torch.device("cpu")
            self.logger.info("Operation will be on *****CPU***** ")

        self.ce_loss_weight = torch.FloatTensor([0.58872014284134, 3.2375309467316, 1.0122526884079]).to(self.device)
        self.writer = SummaryWriter(log_dir=self.config.summary_dir, comment='SHM')

    def save_checkpoint(self, filename=None):
        state = {
            'epoch': self.current_epoch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }
        if filename is None:
            filename = 'checkpoint-epoch{}.pth'.format(self.current_epoch)
        torch.save(state, os.path.join(self.config.checkpoint_dir, filename))

    def load_checkpoint(self):
        try:
            if self.config.mode == 'pretrain_tnet':
                if self.config.tnet_checkpoint is not None:
                    filename = os.path.join(self.config.checkpoint_dir, self.config.tnet_checkpoint)
                    checkpoint = torch.load(filename)
                    self.current_epoch = checkpoint['epoch']
                    model_to_load = {k.replace('module.', ''): v for k, v in checkpoint['state_dict'].items()}
                    self.model.load_state_dict(model_to_load)
                    self.model.to(self.device)
                    self.optimizer.load_state_dict(checkpoint['optimizer'])
            elif self.config.mode == 'pretrain_mnet':
                if self.config.mnet_checkpoint is not None:
                    filename = os.path.join(self.config.checkpoint_dir, self.config.mnet_checkpoint)
                    checkpoint = torch.load(filename)
                    self.current_epoch = checkpoint['epoch']
                    self.model.load_state_dict(checkpoint['state_dict'])
                    self.model.to(self.device)
                    self.optimizer.load_state_dict(checkpoint['optimizer'])
            elif self.config.mode == 'end_to_end':
                if self.config.shm_checkpoint is not None:
                    filename = os.path.join(self.config.checkpoint_dir, self.config.shm_checkpoint)
                    checkpoint = torch.load(filename)
                    self.current_epoch = checkpoint['epoch']
                    self.model.load_state_dict(checkpoint['state_dict'])
                    self.model.to(self.device)
                    self.optimizer.load_state_dict(checkpoint['optimizer'])
                    return
                if self.config.tnet_checkpoint is not None:
                    filename = os.path.join(self.config.savedir,'experiment', 'tnet', 'checkpoints', self.config.tnet_checkpoint)
                    checkpoint = torch.load(filename)
                    self.model.tnet.load_state_dict(checkpoint['state_dict'])
                if self.config.mnet_checkpoint is not None:
                    filename = os.path.join(self.config.savedir,'experiment', 'mnet', 'checkpoints', self.config.mnet_checkpoint)
                    checkpoint = torch.load(filename)
                    self.model.mnet.load_state_dict(checkpoint['state_dict'])
            elif self.config.mode == 'test':
                if self.config.shm_checkpoint is not None:
                    filename = os.path.join(self.config.checkpoint_dir, self.config.shm_checkpoint)
                    checkpoint = torch.load(filename)
                    self.current_epoch = checkpoint['epoch']
                    self.model.load_state_dict(checkpoint['state_dict'])
                    return
                if self.config.tnet_checkpoint is not None:
                    filename = os.path.join(self.config.savedir,'experiment', 'tnet', 'checkpoints', self.config.tnet_checkpoint)
                    checkpoint = torch.load(filename)
                    self.model.tnet.load_state_dict(checkpoint['state_dict'])
                if self.config.mnet_checkpoint is not None:
                    filename = os.path.join(self.config.savedir,'experiment', 'mnet', 'checkpoints', self.config.mnet_checkpoint)
                    checkpoint = torch.load(filename)
                    self.model.mnet.load_state_dict(checkpoint['state_dict'])
        except OSError as e:
            self.logger.info("No checkpoint exists. Skipping...")
            self.logger.info("**First time to train**")

    def trimap_to_image(self, trimap):
        n, c, h, w = trimap.size()
        if c == 3:
            trimap = torch.argmax(trimap, dim=1, keepdim=False)
        return trimap.float().div_(2.0).view(n, 1, h, w)

    def alpha_to_image(self, alpha):
        return alpha.clamp_(0, 1)

    def run(self):
        assert self.config.mode in ['pretrain_tnet', 'pretrain_mnet', 'end_to_end', 'test']
        try:
            if self.config.mode == 'pretrain_tnet':
                self.train_tnet()
            elif self.config.mode == 'pretrain_mnet':
                self.train_mnet()
            elif self.config.mode == 'end_to_end':
                self.train_end_to_end()
            else:
                self.test()
        except KeyboardInterrupt:
            self.logger.info('You have entered CTRL+C.. Wait to finalize')
            self.finalize()


    def train_tnet(self):
        self.model = self.model.tnet
        self.loss_t = ClassificationLoss(w=self.ce_loss_weight)
        self.optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()),
                                    lr=self.config.lr, betas=(0.9, 0.999),
                                    weight_decay=self.config.weight_decay)
        self.load_checkpoint()

        self.model.to(self.device)
        self.loss_t.to(self.device)
        if self.cuda and self.config.ngpu > 1:
            self.model = nn.DataParallel(self.model, device_ids=list(range(self.config.ngpu)))

        sample_image, sample_trimap_gt, _ = make_sample(self.config.mode)
        
        for epoch in range(self.current_epoch, self.config.max_epoch):
            
            self.model.train()
            ########################## Train ############################
            loss_t_epoch = AverageMeter()
            acc_t_epoch = AverageMeter()
            tqdm_loader = tqdm(self.data_loader.train_loader,
                               total=self.data_loader.train_iterations,
                               desc="Train epoch-{}-".format(self.current_epoch + 1))

            for image, trimap_gt, _ in tqdm_loader:
                image, trimap_gt = image.to(self.device), trimap_gt.to(self.device)
                self.optimizer.zero_grad()
                trimap_pre = self.model(image)
                loss_t = self.loss_t(trimap_pre, trimap_gt)
                acc = accuracy(trimap_pre, trimap_gt)
                loss_t.backward()
                self.optimizer.step()

                acc_t_epoch.update(round(acc, 3))
                loss_t_epoch.update(round(loss_t.item(), 3))
                desc = 'Train epoch-{}/ {}: loss {} , acc {}||'.format(
                    self.current_epoch + 1, self.config.max_epoch, loss_t_epoch.val, acc_t_epoch.val)
                tqdm_loader.set_description(desc)
            self.writer.add_scalar('pretrain_tnet/loss_classification', loss_t_epoch.val, self.current_epoch+1)
            self.writer.add_scalar('pretrain_tnet/accuracy', acc_t_epoch.val, self.current_epoch+1)
            ########################## Eval ############################
            if self.config.eval:
                self.model.eval()
                loss_v_epoch = AverageMeter()
                acc_v_epoch = AverageMeter()
                tqdm_loader = tqdm(self.eval_loader.train_loader,
                                   total=self.eval_loader.train_iterations,
                                    desc="Validate epoch-{}-".format(self.current_epoch + 1))
                for image, trimap_gt, _ in tqdm_loader:
                    image, trimap_gt = image.to(
                        self.device), trimap_gt.to(self.device)
                    #self.optimizer.zero_grad()
                    with torch.no_grad():
                        trimap_pre = self.model(image)
                    loss_t = self.loss_t(trimap_pre, trimap_gt)
                    acc = accuracy(trimap_pre, trimap_gt)
                    loss_v_epoch.update(round( loss_t.item(), 3 ))
                    acc_v_epoch.update(round(acc, 3))
                    desc = 'Validate epoch-{}/ {}: loss {} , acc {} ||'.format(
                        self.current_epoch + 1, self.config.max_epoch, loss_v_epoch.val, acc_v_epoch.val)
                    tqdm_loader.set_description(desc)
                self.writer.add_scalar(
                    'pretrain_tnet/eval_loss_classification', loss_v_epoch.val, self.current_epoch + 1)
                self.writer.add_scalar(
                    'pretrain_tnet/eval_accuracy', acc_v_epoch.val, self.current_epoch + 1)
            
            self.current_epoch += 1
            if self.current_epoch % self.config.sample_period == 0:
                self.model.eval()
                with torch.no_grad():
                    sample_image = sample_image.to(self.device)
                    sample_trimap_pre = self.model(sample_image)

                    loss_t_ = self.loss_t(sample_trimap_pre, sample_trimap_gt.to(self.device))
                    acc_t_ = accuracy(sample_trimap_pre, sample_trimap_gt.to(self.device))
                    print(f'Test sample: loss={round(loss_t_.item(), 3)}, acc : {round(acc_t_, 3)}')
                    
                    sample_trimap_pre = self.trimap_to_image(sample_trimap_pre.cpu())
                    self.writer.add_image('pretrain_tnet/sample_trimap_prediction',
                                          make_grid(sample_trimap_pre, nrow=1),
                                          self.current_epoch)
                    save_image(sample_trimap_pre,
                               os.path.join(self.config.out_dir, 'sample_trimap_{}.png'.format(self.current_epoch)),
                               nrow=1, padding=0)
            if self.current_epoch % self.config.checkpoint_period == 0:
                self.save_checkpoint(filename=self.config.checkpoint_name)
            print("Training Results at epoch-" + str(self.current_epoch) + " | " +
                  "loss_classification: " + str(loss_t_epoch.val))
            print('#'*16)

    def train_mnet(self):
        self.model = self.model.mnet
        self.loss_p = PredictionL1Loss()
        self.optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()),
                                    lr=self.config.lr, betas=(0.9, 0.999),
                                    weight_decay=self.config.weight_decay)
        self.load_checkpoint()

        self.model.to(self.device)
        self.loss_p.to(self.device)
        if self.cuda and self.config.ngpu > 1:
            self.model = nn.DataParallel(self.model, device_ids=list(range(self.config.ngpu)))

        sample_image, sample_trimap_gt, sample_alpha_gt = make_sample(self.config.mode)
        sample_trimap_gt = sample_trimap_gt.float()
        for epoch in range(self.current_epoch, self.config.max_epoch):
            
            self.model.train()
            ########################## Train ############################
            loss_p_epoch = AverageMeter()
            loss_alpha_epoch = AverageMeter()
            loss_comps_epoch = AverageMeter()
            acc_t_epoch = AverageMeter()

            tqdm_loader = tqdm(self.data_loader.train_loader,
                               total=self.data_loader.train_iterations,
                               desc="Train epoch-{}-".format(self.current_epoch + 1))
            for image, trimap_gt, alpha_gt in tqdm_loader:
                image, trimap_gt, alpha_gt = image.to(self.device), trimap_gt.float().to(self.device), alpha_gt.to(self.device)

                input = torch.cat([image, trimap_gt], dim=1)

                self.optimizer.zero_grad()
                alpha_pre = self.model(input)
                loss_p, loss_alpha, loss_comps = self.loss_p(image, alpha_pre, alpha_gt)
                acc = round(accuracy((alpha_pre*255.).floor(), (255.0*alpha_gt).floor()), 3)

                loss_p.backward()
                self.optimizer.step()

                loss_p_epoch.update(round(loss_p.item(), 3))
                loss_alpha_epoch.update(round(loss_alpha.item(), 3))
                loss_comps_epoch.update(round(loss_comps.item(), 3))
                acc_t_epoch.update(acc)
                desc = 'Train epoch-{}/ {}: lossp {}, loss alpha {}, loss comps {}, acc {} ||'.format(
                                        self.current_epoch + 1, 
                                        self.config.max_epoch, 
                                        loss_p_epoch.val,
                                        loss_alpha_epoch.val,
                                        loss_comps_epoch.val,
                                        acc_t_epoch.val,
                                )
                tqdm_loader.set_description(desc)

            self.writer.add_scalar(
                'pretrain_mnet/loss_prediction', loss_p_epoch.val, self.current_epoch+1)
            self.writer.add_scalar(
                'pretrain_mnet/loss_alpha_prediction', loss_alpha_epoch.val, self.current_epoch+1)
            self.writer.add_scalar(
                'pretrain_mnet/loss_composition', loss_comps_epoch.val, self.current_epoch+1)
            self.writer.add_scalar(
                'pretrain_mnet/accuracy', acc_t_epoch.val, self.current_epoch+1)
            
            ########################## Eval ############################
            if self.config.eval:
                self.model.eval()
                
                vloss_p_epoch = AverageMeter()
                vloss_alpha_epoch = AverageMeter()
                vloss_comps_epoch = AverageMeter()
                acc_v_epoch = AverageMeter()
                vtqdm_loader = tqdm(self.eval_loader.train_loader,
                                   total=self.eval_loader.train_iterations,
                                   desc="Validate epoch-{}-".format(self.current_epoch + 1))
                for image, trimap_gt, alpha_gt in vtqdm_loader:

                    image, trimap_gt, alpha_gt = image.to(self.device), trimap_gt.float().to(self.device), alpha_gt.to(self.device)
                    input = torch.cat([image, trimap_gt], dim=1)

                    with torch.no_grad():
                        alpha_pre = self.model(input)

                    acc = round(accuracy(alpha_pre, alpha_gt), 3)
                    loss_p, loss_alpha, loss_comps = self.loss_p(image, alpha_pre, alpha_gt)
                    
                    vloss_p_epoch.update(round(loss_p.item(), 3))
                    vloss_alpha_epoch.update(round(loss_alpha.item(), 3))
                    vloss_comps_epoch.update(round(loss_comps.item(), 3))
                    acc_v_epoch.update(acc)
                    desc = 'Validate epoch-{}/ {}: lossp {}, loss alpha {}, loss comps {}, acc {} ||'.format(
                                        self.current_epoch + 1, 
                                        self.config.max_epoch, 
                                        vloss_p_epoch.val,
                                        vloss_alpha_epoch.val,
                                        vloss_comps_epoch.val,
                                        acc_v_epoch.val
                                )
                    vtqdm_loader.set_description(desc)

                self.writer.add_scalar(
                    'pretrain_mnet/eval_loss_prediction', vloss_p_epoch.val, self.current_epoch + 1)
                self.writer.add_scalar(
                    'pretrain_mnet/eval_loss_alpha_prediction', vloss_alpha_epoch.val, self.current_epoch + 1)
                self.writer.add_scalar(
                    'pretrain_mnet/eval_loss_composition', vloss_comps_epoch.val, self.current_epoch + 1)
                self.writer.add_scalar(
                    'pretrain_mnet/eval_accuracy', acc_v_epoch.val, self.current_epoch + 1)

            self.current_epoch += 1
            
            
            if self.current_epoch % self.config.sample_period == 0:
                self.model.eval()
                with torch.no_grad():
                    sample_input = torch.cat((sample_image, sample_trimap_gt), dim=1)
                    sample_input = sample_input.to(self.device)
                    sample_alpha_pre = self.model(sample_input)

                    _loss_p, _loss_alpha, _loss_comps = self.loss_p(
                        sample_image.to(self.device), 
                        sample_alpha_pre, 
                        sample_alpha_gt.to(self.device)
                        )
                    acc_t_ = accuracy(sample_alpha_pre,
                                      sample_alpha_gt.to(self.device))
                    print(
                        f'Test sample: lossp= {round(_loss_p.item(), 3)}, loss_alpha= {round(_loss_alpha.item(), 3)}, loss_comps= {round(_loss_comps.item(), 3)}, acc= {round(acc_t_, 3)}')
                    sample_alpha_pre = self.alpha_to_image(sample_alpha_pre.cpu())

                    self.writer.add_image('pretrain_mnet/sample_alpha_prediction',
                                          make_grid(sample_alpha_pre, nrow=1),
                                          self.current_epoch)
                    save_image(sample_alpha_pre,
                               os.path.join(self.config.out_dir, 'sample_alpha_{}.png'.format(self.current_epoch)),
                               nrow=1, padding=0)
            if self.current_epoch % self.config.checkpoint_period == 0:
                self.save_checkpoint(filename=self.config.checkpoint_name)
            print("Training Results at epoch-" + str(self.current_epoch) + " | " +
                  "loss_prediction: " + str(loss_p_epoch.val) + " loss_alpha_prediction: " +
                  str(loss_alpha_epoch.val) + " loss_composition: " + str(loss_comps_epoch.val))
            print('#'*16)

    def train_end_to_end(self):
        self.loss_p = PredictionL1Loss()
        self.loss_t = ClassificationLoss()
        self.optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()),
                                    lr=self.config.lr, betas=(0.9, 0.999),
                                    weight_decay=self.config.weight_decay)
        self.load_checkpoint()

        self.model.to(self.device)
        self.loss_p.to(self.device)
        self.loss_t.to(self.device)
        if self.cuda and self.config.ngpu > 1:
            self.model = nn.DataParallel(self.model, device_ids=list(range(self.config.ngpu)))

        sample_image, sample_trimap_gt, sample_alpha_gt = make_sample(self.config.mode)
        ########################## Train ############################
        for epoch in range(self.current_epoch, self.config.max_epoch):
            self.model.train()
            loss_epoch = AverageMeter()
            loss_p_epoch = AverageMeter()
            loss_alpha_epoch = AverageMeter()
            loss_comps_epoch = AverageMeter()
            loss_t_epoch = AverageMeter()
            acc_alp_epoch = AverageMeter()
            acc_tri_epoch = AverageMeter()
            tqdm_loader = tqdm(self.data_loader.train_loader,
                               total=self.data_loader.train_iterations,
                               desc="Train epoch-{}-".format(self.current_epoch+1))
            for image, trimap_gt, alpha_gt in tqdm_loader:
                image, trimap_gt, alpha_gt = image.to(self.device), trimap_gt.to(self.device), alpha_gt.to(self.device)
                self.optimizer.zero_grad()
                trimap_pre, alpha_pre = self.model(image)
                loss_p, loss_alpha, loss_comps = self.loss_p(image, alpha_pre, alpha_gt)
                loss_t = self.loss_t(trimap_pre, trimap_gt)
                loss = loss_p + self.config.loss_lambda * loss_t
                acc_tri = round(accuracy(trimap_pre, trimap_gt), 3)
                acc_alp = round(accuracy(alpha_pre, alpha_gt), 3)
                loss.backward()
                self.optimizer.step()

                loss_epoch.update(loss.item())
                loss_p_epoch.update(loss_p.item())
                loss_alpha_epoch.update(loss_alpha.item())
                loss_comps_epoch.update(loss_comps.item())
                loss_t_epoch.update(loss_t.item())

                acc_tri_epoch.update(acc_tri)
                acc_alp_epoch.update(acc_alp)
                desc = 'Train epoch-{}/ {}: loss {}, lossp {}, loss alpha {}, loss comps {}, loss t {}, acc_tri {}, acc_alp {} ||'.format(
                    self.current_epoch + 1,
                    self.config.max_epoch,
                    loss_epoch.val,
                    loss_p_epoch.val,
                    loss_alpha_epoch.val,
                    loss_comps_epoch.val,
                    loss_t_epoch.val,
                    acc_tri_epoch.val,
                    acc_alp_epoch.val
                )
                tqdm_loader.set_description(desc)

            self.writer.add_scalar('end_to_end/loss', loss_epoch.val, self.current_epoch + 1)
            self.writer.add_scalar('end_to_end/loss_prediction', loss_p_epoch.val, self.current_epoch + 1)
            self.writer.add_scalar('end_eo_end/loss_alpha_prediction', loss_alpha_epoch.val, self.current_epoch + 1)
            self.writer.add_scalar('end_to_end/loss_composition', loss_comps_epoch.val, self.current_epoch + 1)
            self.writer.add_scalar('end_to_end/loss_classification', loss_t_epoch.val, self.current_epoch + 1)
            self.writer.add_scalar('end_to_end/acc_tri', acc_tri_epoch.val, self.current_epoch + 1)
            self.writer.add_scalar('end_to_end/acc_alp', acc_alp_epoch.val, self.current_epoch + 1)
            
            ########################## Eval ############################
            if self.config.eval:
                self.model.eval()

                vloss_epoch = AverageMeter()
                vloss_p_epoch = AverageMeter()
                vloss_alpha_epoch = AverageMeter()
                vloss_comps_epoch = AverageMeter()
                vloss_t_epoch = AverageMeter()
                acc_alp_epoch = AverageMeter()
                acc_tri_epoch = AverageMeter()
                tqdm_loader = tqdm(self.eval_loader.train_loader,
                                   total=self.eval_loader.train_iterations,
                                   desc="Validate epoch-{}-".format(self.current_epoch + 1))

                for image, trimap_gt, alpha_gt in tqdm_loader:
                    image, trimap_gt, alpha_gt = image.to(self.device), trimap_gt.to(self.device), alpha_gt.to(self.device)
                    #
                    trimap_pre, alpha_pre = self.model(image)
                    loss_p, loss_alpha, loss_comps = self.loss_p(image, alpha_pre, alpha_gt)
                    loss_t = self.loss_t(trimap_pre, trimap_gt)
                    loss = loss_p + self.config.loss_lambda * loss_t

                    acc_tri = round(accuracy(trimap_pre, trimap_gt), 3)
                    acc_alp = round(accuracy(alpha_pre, alpha_gt), 3)
                    
                    vloss_epoch.update(loss.item())
                    vloss_p_epoch.update(loss_p.item())
                    vloss_alpha_epoch.update(loss_alpha.item())
                    vloss_comps_epoch.update(loss_comps.item())
                    vloss_t_epoch.update(loss_t.item())
                    
                    acc_tri_epoch.update(acc_tri)
                    acc_alp_epoch.update(acc_alp)
                    desc = 'Train epoch-{}/ {}: loss {}, lossp {}, loss alpha {}, loss comps {}, loss t {}, acc_tri {}, acc_alp {} ||'.format(
                        self.current_epoch + 1,
                        self.config.max_epoch,
                        vloss_epoch.val,
                        vloss_p_epoch.val,
                        vloss_alpha_epoch.val,
                        vloss_comps_epoch.val,
                        vloss_t_epoch.val,
                        acc_tri_epoch.val,
                        acc_alp_epoch.val
                    )
                    tqdm_loader.set_description(desc)

                self.writer.add_scalar(
                    'end_to_end/eval_loss', vloss_epoch.val, self.current_epoch + 1)
                self.writer.add_scalar(
                    'end_to_end/eval_loss_prediction', vloss_p_epoch.val, self.current_epoch + 1)
                self.writer.add_scalar(
                    'end_eo_end/eval_loss_alpha_prediction', vloss_alpha_epoch.val, self.current_epoch + 1)
                self.writer.add_scalar(
                    'end_to_end/eval_loss_composition', vloss_comps_epoch.val, self.current_epoch + 1)
                self.writer.add_scalar(
                    'end_to_end/eval_loss_classification', vloss_t_epoch.val, self.current_epoch + 1)
                self.writer.add_scalar(
                    'end_to_end/eval_acc_tri', acc_tri_epoch.val, self.current_epoch +1)
                self.writer.add_scalar('end_to_end/eval_acc_alp',
                                   acc_alp_epoch.val, self.current_epoch +1)
            
            self.current_epoch += 1
            if self.current_epoch % self.config.sample_period == 0:
                self.model.eval()
                with torch.no_grad():
                    sample_image = sample_image.to(self.device)
                    sample_trimap_gt = sample_trimap_gt.to(self.device)
                    sample_alpha_gt = sample_alpha_gt.to(self.device)

                    sample_trimap_pre, sample_alpha_pre = self.model(sample_image)
                    loss_p, loss_alpha, loss_comps = self.loss_p(
                        sample_image, sample_alpha_pre, sample_alpha_gt)
                    loss_t = self.loss_t(sample_trimap_pre, sample_trimap_gt)
                    loss = loss_p + self.config.loss_lambda * loss_t
                    
                    acc_tri = round(
                        accuracy(sample_trimap_pre, sample_trimap_gt), 3)
                    acc_alp = round(
                        accuracy(sample_alpha_pre, sample_alpha_gt), 3)
                    print(
                        f'Test sample: loss= {round(loss.item(), 3)}, lossp= {round(loss_p.item(), 3)}, \
                        loss_alpha= {round(loss_alpha.item(), 3)}, \
                        loss_comps= {round(loss_comps.item(), 3)}, \
                        acc_tri= {round(acc_tri, 3)} \
                        acc_alp= {round(acc_alp, 3)}')
                    sample_trimap_pre = self.trimap_to_image(sample_trimap_pre.cpu())
                    sample_alpha_pre = self.alpha_to_image(sample_alpha_pre.cpu())
                    self.writer.add_image('sample_trimap_prediction',
                                          make_grid(sample_trimap_pre, nrow=1),
                                          self.current_epoch)
                    self.writer.add_image('sample_alpha_prediction',
                                          make_grid(sample_alpha_pre, nrow=1),
                                          self.current_epoch)
                    save_image(sample_trimap_pre,
                               os.path.join(self.config.out_dir, 'sample_trimap_{}.png'.format(self.current_epoch)),
                               nrow=1, padding=0)
                    save_image(sample_alpha_pre,
                               os.path.join(self.config.out_dir, 'sample_alpha_{}.png'.format(self.current_epoch)),
                               nrow=1, padding=0)
            if self.current_epoch % self.config.checkpoint_period == 0:
                self.save_checkpoint(filename=self.config.checkpoint_name)
            print("Training Results at epoch-" + str(self.current_epoch) + " | " +
                  "loss: " + str(loss_epoch.val) + " loss_prediction: " + str(loss_p_epoch.val) +
                  " loss_alpha_prediction: " + str(loss_alpha_epoch.val) + " loss_composition: " +
                  str(loss_comps_epoch.val) + " loss_classification: " + str(loss_t_epoch.val))
            print('#'*16)
    def test(self):
        self.load_checkpoint()

        self.model = self.model.to(self.device)

        tqdm_loader = tqdm(self.data_loader.test_loader, total=self.data_loader.test_iterations,
                          desc="Testing at checkpoint")

        self.model.eval()
        for image_name, image, trimap_gt, alpha_gt in tqdm_loader:
            batch_size = len(image_name)
            image = image.to(self.device)
            with torch.no_grad():
                trimap_pre, alpha_pre = self.model(image)
                for i in range(batch_size):
                    save_image(trimap_pre[i].cpu(),
                               os.path.join(self.config.out_dir, '{}_trimap.png'.format(image_name[i][:-4])),
                               nrow=1, padding=0)
                    save_image(alpha_pre[i].cpu(),
                               os.path.join(self.config.out_dir, '{}_alpha.png'.format(image_name[i][:-4])),
                               nrow=1, padding=0)
        print('Test finished')

    def finalize(self):
        print('Please wait while finalizing the operation.. Thank you')
        self.save_checkpoint('checkpoint-suspend.pth.tar')
        self.writer.export_scalars_to_json(os.path.join(self.config.summary_dir, 'all_scalars.json'))
        self.writer.close()
        self.data_loader.finalize()
