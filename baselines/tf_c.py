import torch
from tqdm import tqdm
from src.models.tf_cencoder import *
from src.src_utils.augmentations import DataTransform_FD, DataTransform_TD
from src.models.common import Seq_Transformer

from pytorch_lightning.loggers import WandbLogger
import wandb
from statsmodels.tsa.stattools import adfuller
import numpy as np
import math
from src.losses.contrastive import NTXentLoss, NTXentLoss_poly
import torch.fft as fft
from src.src_utils.utils import cosine_warmup_scheduler
import time
from utils import name_with_datetime
import os


class TF_C:
    '''The TF_C model'''
    
    def __init__(
        self,
        args,
        config,
        device='cuda',
    ):
        '''
          Initialize a TF_C model.

        '''
        
        self.args = args
        self.config = config
        super().__init__()
        
        self.device = device

        # self.net = FeatureProjector(input_size=args['feature_dim'], output_size=args['out_features']).to(self.device)
        self.net = TFC(input_dims=args['feature_dim'], output_dims=args['out_features']).to(self.device)
        
        self.n_iters = 0

    def fit(self, train_dataset, ds_name, verbose=False):
        ''' Training the TF_C model.
        
        Args:
            train_data (numpy.ndarray): The training data. It should have a shape of (n_instance, n_timestamps, n_features). All missing data should be set to NaN.
            n_epochs (Union[int, NoneType]): The number of epochs. When this reaches, the training stops.
            n_iters (Union[int, NoneType]): The number of iterations. When this reaches, the training stops. If both n_epochs and n_iters are not specified, a default setting would be used that sets n_iters to 200 for a dataset with size <= 100000, 600 otherwise.
            verbose (bool): Whether to print the training loss after each epoch.
            
        Returns:
            loss_log: a list containing the training losses on each epoch.
        '''
        
        train_loader = torch.utils.data.DataLoader(
                dataset=train_dataset,
                batch_size= self.args['batch_size'],
                shuffle = True,
                num_workers=self.config.NUM_WORKERS,
                drop_last = True,
            )
        
        # Wandb setup
        if self.config.WANDB:    
            proj_name = 'Dynamic_CL' + ds_name + str(self.config.SEED)
            run_name = 'TF_C'

            wandb_logger = WandbLogger(project=proj_name)
            
            # Initialize Wandb
            wandb.init(project=proj_name, name=run_name)
            wandb.watch(self.net, log='all', log_freq=100)

            # Update Wandb config
        
            wandb.config.update(self.args)
            wandb.config.update({
                'Algorithm': f'{run_name}',
                'Dataset': f'{ds_name}',
                'Train_DS_size': len(train_dataset),
                'Batch_Size': self.args["batch_size"],
                'Epochs': self.args["epochs"],
                'Patience': self.config.PATIENCE,
                'Seed': self.config.SEED

            })
            wandb.run.name = run_name
            wandb.run.save()

        self.args['lr'] = float(self.args['lr'])
        self.args['weight_decay'] = float(self.args['weight_decay'])

        optimizer = torch.optim.AdamW(self.net.parameters(), lr=self.args['lr'], betas=(0.9, 0.99), weight_decay=self.args['weight_decay'])

        n_iters = self.args['iterations']
        pbar = tqdm(total=n_iters, desc="Training")
        epoch = 0
        num_training_steps = n_iters
        num_warmup_steps = int(0.1 * n_iters)

        scheduler = cosine_warmup_scheduler(optimizer, num_warmup_steps, num_training_steps)

        if self.args['save_model']:
            run_dir = f'results/{ds_name}/seed_{self.config.SEED}/{name_with_datetime(self.__class__.__name__)}'
            os.makedirs(run_dir, exist_ok=True)
            start_time = time.time()

        while True:

            # Training phase
            self.net.train()  # Set the model to training mode
            train_running_loss = 0.0
            n_epoch_iters = 0

            for x, _ in train_loader:

                interrupted = False
                if n_iters is not None and self.n_iters >= n_iters:
                    interrupted = True
                    break
                
                B, _, _ = x.shape

                x_f = fft.fft(x).abs() #/(window_length) # rfft for real value inputs.
               
                aug1 = DataTransform_TD(x)
                aug1_f = DataTransform_FD(x_f)

                data = x.float().to(self.device)
                aug1 = aug1.float().to(self.device)
                data_f, aug1_f = x_f.float().to(self.device), aug1_f.float().to(self.device)

                
                """Produce embeddings"""
                h_t, z_t, h_f, z_f = self.net(data, data_f)
                h_t_aug, z_t_aug, h_f_aug, z_f_aug = self.net(aug1, aug1_f)

                """Compute Pre-train loss"""
                """NTXentLoss: normalized temperature-scaled cross entropy loss. From SimCLR"""
                nt_xent_criterion = NTXentLoss(self.device, B, 0.2,
                                            True)
                
                loss_t = nt_xent_criterion(h_t, h_t_aug)
                loss_f = nt_xent_criterion(h_f, h_f_aug)
                l_TF = nt_xent_criterion(z_t, z_f) # this is the initial version of TF loss

                l_1, l_2, l_3 = nt_xent_criterion(z_t, z_f_aug), nt_xent_criterion(z_t_aug, z_f), nt_xent_criterion(z_t_aug, z_f_aug)
                loss_c = (1 + l_TF - l_1) + (1 + l_TF - l_2) + (1 + l_TF - l_3)

                lam = 0.2
                loss = lam*(loss_t + loss_f) + l_TF
                
                # Backward pass and optimization
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                    
                # Update training statistics
                n_epoch_iters += 1
                self.n_iters += 1
                pbar.update(1)

                train_running_loss += loss.item()

            scheduler.step()
            if interrupted:
                break
            train_running_loss /= n_epoch_iters
    
            if verbose:
                print(f"Epoch {epoch}, Train Loss: {train_running_loss:.4f}")

            # Log training loss to Wandb
            if self.config.WANDB:
                wandb.log({'Train Loss': train_running_loss, 'Epoch': epoch})

        # Save model
        if self.args['save_model']:
            model_path = os.path.join(run_dir, f'model.pt')
            torch.save(self.net.state_dict(), model_path)

            total_time = time.time() - start_time

            # Save training time
            time_file = os.path.join(run_dir, 'time.txt')
            with open(time_file, 'w') as f:
                f.write(str(total_time))
            
        try:   
            return train_running_loss
        except:
            return 0

    def encode(self, x):
        self.net.eval()
        out = self.net.feature_extractor_t(x.to(self.device))

        return out


    def save(self, fn):
        ''' Save the model to a file.
        
        Args:
            fn (str): filename.
        '''
        torch.save(self.net.state_dict(), fn)
    
    def load(self, fn):
        ''' Load the model from a file.
        
        Args:
            fn (str): filename.
        '''
        state_dict = torch.load(fn, map_location=self.device)
        self.net.load_state_dict(state_dict)
