import logging
import os
import sys

import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from utils import utils
from models.simclr_model import SimCLRModel
from loss.nt_xent import NTXentLoss
import numpy as np


torch.manual_seed(0)


class SimCLR(object):

    def __init__(self, config):
        self.config  = config
        self.device  = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.writer  = SummaryWriter(log_dir=config["log_dir"])
        self.train_dataset, self.valid_dataset = utils.data_loader(config)

        self.model     = SimCLRModel(base_encoder=config["base_encoder"], dim=config["out_dim"]).to(self.device)
        self.criterion = NTXentLoss(self.device, config['batch_size'],
                                config['temperature'], config['use_cosine'])
        self.optimizer = torch.optim.Adam(self.model.parameters(), config["learning_rate"],
                                        weight_decay=config["weight_decay"])
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer,
                                        T_max=len(self.train_dataset), eta_min=0,
                                        last_epoch=-1)

        # utils.visualize_dataset(self.train_dataset)
        self.checkpoints_folder = config["checkpoints"]
        self.config['eval_every_n_steps'] = self.config['eval_every_n_steps'] if self.config['eval_every_n_steps']!=-1 else len(self.train_dataset)
        self.config['log_every_n_steps'] = self.config['log_every_n_steps'] if self.config['log_every_n_steps']!=-1 else len(self.train_dataset)
        logging.basicConfig(filename=os.path.join(self.writer.log_dir, 'training.log'), level=logging.INFO)
        logging.StreamHandler(sys.stdout)
        # logging.addHandler()

        self._load_pre_trained_weights()

    def _load_pre_trained_weights(self):
        try:
            state_dict = torch.load(os.path.join(self.checkpoints_folder, 'model.pth'))
            self.model.load_state_dict(state_dict)
            logging.info("Loaded pre-trained model with success.")
        except FileNotFoundError:
            logging.info("Pre-trained weights not found. Training from scratch.")

    def _step(self, xis, xjs):
        # get the representations and the projections
        _, zis = self.model(xis)  # [N,C]
        # get the representations and the projections
        _, zjs = self.model(xjs)  # [N,C]
        # normalize projection feature vectors
        zis = F.normalize(zis, dim=1)
        zjs = F.normalize(zjs, dim=1)

        loss = self.criterion(zis, zjs)
        return loss

    def _validate(self):
        # validation steps
        with torch.no_grad():
            self.model.eval()

            valid_loss = 0.0
            counter = 0
            prefix = f'Validation '
            for data in tqdm(self.valid_dataset, desc=prefix,
                    dynamic_ncols=True, leave=True, position=0):

                _, xis, xjs, _ = data
                xis = xis.to(self.device)
                xjs = xjs.to(self.device)

                loss = self._step(xis, xjs)
                valid_loss += loss.item()
                counter += 1
            valid_loss /= counter
        self.model.train()
        return valid_loss

    def train(self):

        logging.info(f"Start SimCLR training for {self.config['epochs']} epochs.")
        logging.info(f"Training with device: {self.device}.")

        n_iter = 0
        counter = 0
        loss_ = 0
        best_valid_loss = np.inf

        for epoch in range(self.config['epochs']):
            prefix = f'Training Epoch {epoch}: '
            for data in tqdm(self.train_dataset, desc=prefix,
                    dynamic_ncols=True, leave=True, position=0):

                _, xis, xjs, _ = data

                xis = xis.to(self.device)
                xjs = xjs.to(self.device)

                self.optimizer.zero_grad()
                loss = self._step(xis, xjs)

                if n_iter % self.config['log_every_n_steps'] == 0:
                    self.writer.add_scalar('train_loss', loss.item(), global_step=n_iter)

                loss.backward()
                self.optimizer.step()
                n_iter += 1
                counter += 1
                loss_ += loss.item()

            logging.info(f"Training loss at {epoch} epoch, {n_iter} iteration is {loss_/counter}.")
            loss_ = 0
            counter = 0
            # validate the model if requested
            if epoch % self.config['eval_every_n_steps'] == 0:
                valid_loss = self._validate()
                logging.info(f"Validation loss at {epoch} epoch, {n_iter} iteration is {valid_loss}.")
                if valid_loss < best_valid_loss:
                    # save the model weights
                    best_valid_loss = valid_loss
                    torch.save(self.model.state_dict(), os.path.join(self.checkpoints_folder, f"model.pth"))
                    logging.info(f"Saved model weights at {epoch} epoch, {n_iter} iteration.")

                self.writer.add_scalar('valid_loss', valid_loss, global_step=n_iter)

            # warmup for the first 10 epochs
            if epoch >= 10:
                self.scheduler.step()
            self.writer.add_scalar('cosine_lr_decay', self.scheduler.get_lr()[0], global_step=n_iter)

        logging.info("Training has finished.")
        logging.info(f"Model checkpoint and metadata has been saved at {self.writer.log_dir}.")
