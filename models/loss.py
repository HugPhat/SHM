from itertools import combinations
import torch
import torch.nn as nn
import torch.nn.functional as F


class PredictionL1Loss(nn.Module):
    def __init__(self, gamma=0.5):
        super(PredictionL1Loss, self).__init__()
        self.gamma = gamma

    def __call__(self, img, alpha_p, alpha_g):
        l_alpha = F.l1_loss(alpha_p, alpha_g, reduction='mean')*5

        fg_p = alpha_p.repeat(1, 3, 1, 1) * img
        fg_g = alpha_g.repeat(1, 3, 1, 1) * img
        l_comps = F.l1_loss(fg_p, fg_g, reduction='mean')*5

        #l_p = self.gamma * l_alpha + (1 - self.gamma) * l_comps
        l_p =   l_alpha + l_comps
        return l_p, l_alpha, l_comps


class ClassificationLoss(nn.Module):
    def __init__(self, w=None):
        super(ClassificationLoss, self).__init__()
        self.w= w
    def __call__(self, trimap_p, trimap_g):
        n, _, h, w = trimap_p.size()
        l_t = F.cross_entropy(trimap_p, trimap_g.view(n, h, w).long(), weight=self.w)
        return l_t

############### dice loss #########################

class dice_loss(nn.Module):
    def __init__(self, n_clss=3):
        super(dice_loss, self).__init__()
        self.n_clss = n_clss
        self.eps = 1e-6
    def forward(self, pred, target):
        ch = target.size(1)  # index or one-hot-encoding
        if ch == 1:  # if index-> convert to OHE
            one_hot_target = F.one_hot(target, self.n_clss).permute(
                0, 4, 2, 3, 1).contiguous().squeeze(-1)
        else:
            one_hot_target = target

        one_hot_target = one_hot_target.float().detach() 
        probas = torch.sigmoid(pred.float())

        dims = (0,) + tuple(range(2, target.ndimension()))

        intersection = torch.sum(one_hot_target * probas, dims) + self.eps

        _sum = torch.sum(probas + one_hot_target, dims)

        dice = ((2. * intersection) / (_sum  + self.eps)).mean()

        return 1. - dice

if __name__ == '__main__':
    img = torch.rand(4, 3, 256, 256)
    alpha_p = torch.rand(4, 1, 256, 256)
    alpha_g = torch.rand(4, 1, 256, 256)

    trimap_p = torch.rand(4, 3, 256, 256)
    trimap_g = torch.randint(3, (4, 256, 256), dtype=torch.long)

    Lp = PredictionL1Loss()
    Lc = ClassificationLoss()

    lp = Lp(img, alpha_p, alpha_g)
    print(lp.size(), lp)
    lc = Lc(trimap_p, trimap_g)
    print(lc.size(), lc)
