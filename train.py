import subprocess
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from utils.dataset_utils import PromptTrainDataset
from net.model import PromptIR
from utils.schedulers import LinearWarmupCosineAnnealingLR
import numpy as np
import wandb
from options import options as opt
import lightning.pytorch as pl
from lightning.pytorch.loggers import WandbLogger,TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint


class PromptIRModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.net = PromptIR(decoder=True, prompt_len=opt.prompt_len, num_expert=opt.num_expert)
        self.loss_fn  = nn.L1Loss()
        if opt.use_tur:
            self.uem_head = self._build_uem_head(in_channels=96)
        else:
            self.uem_head = None

    def forward(self,x):
        return self.net(x)

    def _build_uem_head(self, in_channels=96):
        hidden = max(32, in_channels // 2)
        return nn.Sequential(
            nn.Conv2d(in_channels, hidden, kernel_size=3, stride=1, padding=1, bias=True),
            nn.GELU(),
            nn.Conv2d(hidden, hidden, kernel_size=3, stride=1, padding=1, bias=True),
            nn.GELU(),
            nn.Conv2d(hidden, 1, kernel_size=3, stride=1, padding=1, bias=True),
            nn.AdaptiveAvgPool2d(1),
        )

    def compute_uncertainty_weighted_loss(self, base_per_sample, log_var):
        return (0.5 * torch.exp(-log_var) * base_per_sample + 0.5 * opt.lambda_tur * log_var).mean()

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward
        ([clean_name, de_id], degrad_patch, clean_patch) = batch
        restored, aux = self.net(degrad_patch, is_neg=False, return_aux=True)
        rec_loss = self.loss_fn(restored,clean_patch)
        rec_per_sample = torch.abs(restored - clean_patch).mean(dim=(1, 2, 3))

        if opt.use_cpr:
            neg_losses = []
            with torch.no_grad():
                for _ in range(opt.neg_num):
                    neg_restored = self.net(degrad_patch, is_neg=True)
                    neg_losses.append(self.loss_fn(neg_restored, clean_patch))

            neg_loss = torch.stack(neg_losses).mean() if len(neg_losses) > 0 else rec_loss.detach()
            cpr_loss = torch.relu(rec_loss + opt.cpr_margin - neg_loss)
        else:
            cpr_loss = torch.zeros_like(rec_loss)
            neg_loss = torch.zeros_like(rec_loss)

        base_per_sample = rec_per_sample + opt.lambda_cpr * cpr_loss
        base_loss = base_per_sample.mean()

        if opt.use_tur:
            tur_feature = aux.get("tur_feature", None)
            if tur_feature is None:
                raise RuntimeError("Missing TUR feature from model aux output.")
            log_var = self.uem_head(tur_feature).flatten(1).squeeze(1)
            log_var = torch.clamp(log_var, min=-10.0, max=10.0)
            loss = self.compute_uncertainty_weighted_loss(base_per_sample, log_var)
            tur_loss = loss - base_loss
            sigma = torch.exp(0.5 * log_var).mean()
            self.log("train_sigma", sigma)
        else:
            loss = base_loss
            tur_loss = torch.zeros_like(rec_loss)

        self.log("train_rec_loss", rec_loss, prog_bar=True)
        self.log("train_neg_loss", neg_loss)
        self.log("train_cpr_loss", cpr_loss)
        self.log("train_tur_loss", tur_loss)

        # Logging to TensorBoard (if installed) by default
        self.log("train_loss", loss)
        return loss

    def lr_scheduler_step(self,scheduler,metric):
        scheduler.step(self.current_epoch)
        lr = scheduler.get_lr()

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=opt.lr)
        scheduler = LinearWarmupCosineAnnealingLR(optimizer=optimizer,warmup_epochs=15,max_epochs=opt.epochs)

        return [optimizer],[scheduler]






def main():
    print("Options")
    print(opt)
    if opt.wblogger is not None:
        logger  = WandbLogger(project=opt.wblogger,name="PromptIR-Train")
    else:
        logger = TensorBoardLogger(save_dir = "logs/")

    trainset = PromptTrainDataset(opt)
    checkpoint_callback = ModelCheckpoint(dirpath = opt.ckpt_dir,every_n_epochs = 1,save_top_k=-1)
    trainloader = DataLoader(trainset, batch_size=opt.batch_size, pin_memory=True, shuffle=True,
                             drop_last=True, num_workers=opt.num_workers)

    model = PromptIRModel()

    trainer = pl.Trainer( max_epochs=opt.epochs,accelerator="gpu",devices=opt.num_gpus,strategy="ddp_find_unused_parameters_true",logger=logger,callbacks=[checkpoint_callback])
    trainer.fit(model=model, train_dataloaders=trainloader)


if __name__ == '__main__':
    main()



