import torch.nn.functional as F
import torch.nn as nn
import torch

def weights_init(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight.data)
        if m.bias is not None:
            nn.init.zeros_(m.bias.data)

class NeRF(nn.Module):
    def __init__(self, vol_n=8+8, feat_ch=8+16+32+3, hid_n=64):
        super(NeRF, self).__init__()
        self.hid_n = hid_n
        self.agg = Agg(feat_ch)
        self.lr0 = nn.Sequential(nn.Linear(vol_n+16, hid_n), nn.ReLU())
        self.sigma = nn.Sequential(nn.Linear(hid_n, 1), nn.Softplus())
        self.color = nn.Sequential(
            nn.Linear(16+vol_n+feat_ch+hid_n+4, hid_n), # agg_feats+vox_feat+img_feat+lr0_feats+dir
            nn.ReLU(),
            nn.Linear(hid_n, 1)
        )
        self.lr0.apply(weights_init)
        self.sigma.apply(weights_init)
        self.color.apply(weights_init)

    def forward(self, vox_feat, img_feat_rgb_dir, source_img_mask):
        # assert torch.sum(torch.sum(source_img_mask,1)<2)==0
        b, d, n, _ = img_feat_rgb_dir.shape # b,d,n,f=8+16+32+3+4
        agg_feat = self.agg(img_feat_rgb_dir, source_img_mask) # b,d,f=16
        x = self.lr0(torch.cat((vox_feat, agg_feat), dim=-1)) # b,d,f=64
        sigma = self.sigma(x) # b,d,1

        x = torch.cat((x, vox_feat, agg_feat), dim=-1) # b,d,f=16+16+64
        x = x.view(b, d, 1, x.shape[-1]).repeat(1, 1, n, 1)
        x = torch.cat((x, img_feat_rgb_dir), dim=-1)
        logits = self.color(x)
        source_img_mask_ = source_img_mask.reshape(b, 1, n, 1).repeat(1, logits.shape[1], 1, 1) == 0
        logits[source_img_mask_] = -1e7
        color_weight = F.softmax(logits, dim=-2)
        color = torch.sum((img_feat_rgb_dir[..., -7:-4] * color_weight), dim=-2)
        return color, sigma

class Agg(nn.Module):
    def __init__(self, feat_ch):
        super(Agg, self).__init__()
        self.feat_ch = feat_ch
        self.view_fc = nn.Sequential(nn.Linear(4, feat_ch), nn.ReLU())
        self.view_fc.apply(weights_init)
        self.global_fc = nn.Sequential(nn.Linear(feat_ch*3, 32), nn.ReLU())

        self.agg_w_fc = nn.Linear(32, 1)
        self.fc = nn.Linear(32, 16)
        self.global_fc.apply(weights_init)
        self.agg_w_fc.apply(weights_init)
        self.fc.apply(weights_init)

    def masked_mean_var(self, img_feat_rgb, source_img_mask):
        # img_feat_rgb: b,d,n,f   source_img_mask: b,n
        b, n = source_img_mask.shape
        source_img_mask = source_img_mask.view(b, 1, n, 1)
        mean = torch.sum(source_img_mask * img_feat_rgb, dim=-2)/ (torch.sum(source_img_mask, dim=-2) + 1e-5)
        var = torch.sum((img_feat_rgb - mean.unsqueeze(-2)) ** 2 * source_img_mask, dim=-2) / (torch.sum(source_img_mask, dim=-2) + 1e-5)
        return mean, var

    def forward(self, img_feat_rgb_dir, source_img_mask):
        # img_feat_rgb_dir b,d,n,f
        b, d, n, _ = img_feat_rgb_dir.shape
        view_feat = self.view_fc(img_feat_rgb_dir[..., -4:]) # b,d,n,f-4
        img_feat_rgb =  img_feat_rgb_dir[..., :-4] + view_feat

        mean_feat, var_feat = self.masked_mean_var(img_feat_rgb, source_img_mask)
        var_feat = var_feat.view(b, -1, 1, self.feat_ch).repeat(1, 1, n, 1)
        avg_feat = mean_feat.view(b, -1, 1, self.feat_ch).repeat(1, 1, n, 1)

        feat = torch.cat([img_feat_rgb, var_feat, avg_feat], dim=-1) # b,d,n,f
        global_feat = self.global_fc(feat) # b,d,n,f
        logits = self.agg_w_fc(global_feat) # b,d,n,1
        source_img_mask_ = source_img_mask.reshape(b, 1, n, 1).repeat(1, logits.shape[1], 1, 1) == 0
        logits[source_img_mask_] = -1e7
        agg_w = F.softmax(logits, dim=-2)
        im_feat = (global_feat * agg_w).sum(dim=-2)
        return self.fc(im_feat)