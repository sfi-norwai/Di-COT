import torch
from tqdm import tqdm
from src.models.inceptiontime import *
from src.src_utils.utils import DataTransform
from src.models.common import Seq_Transformer

from pytorch_lightning.loggers import WandbLogger
import wandb
from statsmodels.tsa.stattools import adfuller
import numpy as np
import math
from src.losses.contrastive import NTXentLoss
from src.src_utils.utils import cosine_warmup_scheduler
import time
from utils import name_with_datetime
import os


class TC(nn.Module):
    def __init__(self, args, device):
        super(TC, self).__init__()
        self.transformer = Seq_Transformer(patch_size=args['out_features'], dim=args['out_features'], depth=4, heads=4, mlp_dim=64)
        self.device = device
        self.lsoftmax = nn.LogSoftmax()
        self.timestep = 6
        self.Wk = nn.ModuleList([nn.Linear(args['out_features'], args['out_features']) for i in range(self.timestep)])

        self.projection_head = nn.Sequential(
            nn.Linear(args['out_features'], args['out_features'] // 2),
            nn.BatchNorm1d(args['out_features'] // 2),
            nn.ReLU(inplace=True),
            nn.Linear(args['out_features'] // 2, args['out_features'] // 4),
        )

    def forward(self, features_aug1, features_aug2):

        z_aug1 = features_aug1  # features are (batch_size, #channels, seq_len)
        seq_len, feat = z_aug1.shape[1], z_aug1.shape[2]

        z_aug2 = features_aug2
 
        batch = z_aug1.shape[0]
        t_samples = torch.randint(seq_len - self.timestep, size=(1,)).long().to(self.device)  # randomly pick time stamps

        nce = 0  # average over timestep and batch
        encode_samples = torch.empty((self.timestep, batch, feat)).float().to(self.device)

        for i in np.arange(1, self.timestep + 1):
            encode_samples[i - 1] = z_aug2[:, t_samples + i, :].view(batch, feat)
        forward_seq = z_aug1[:, :t_samples + 1, :]

        c_t = self.transformer(forward_seq)

        pred = torch.empty((self.timestep, batch, feat)).float().to(self.device)
        for i in np.arange(0, self.timestep):
            linear = self.Wk[i]
            pred[i] = linear(c_t)
        for i in np.arange(0, self.timestep):
            total = torch.mm(encode_samples[i], torch.transpose(pred[i], 0, 1))
            nce += torch.sum(torch.diag(self.lsoftmax(total)))
        nce /= -1. * batch * self.timestep

        return nce, self.projection_head(c_t)

class TS_TCC:
    '''The TS_TCC model'''
    
    def __init__(
        self,
        args,
        config,
        device='cuda',
    ):
        '''
          Initialize a TS_TCC model.

        '''
        
        self.args = args
        self.config = config
        super().__init__()
        
        self.device = device

        # self.net = FeatureProjector(input_size=args['feature_dim'], output_size=args['out_features']).to(self.device)
        self.net = InceptionTime(n_in_channels=args['feature_dim'], out_channels=args['out_features']).to(self.device)
        self.tc = TC(args, device).to(device)
        
        self.n_iters = 0

    def fit(self, train_dataset, ds_name, verbose=False):
        ''' Training the TS_TCC model.
        
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
            run_name = 'TS_TCC'

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

        optimizer = torch.optim.AdamW([
            {'params': self.net.parameters(), 'lr': self.args['lr']},
            {'params': self.tc.parameters(), 'lr': self.args['lr']},
        ],
        betas=(0.9, 0.999),
        weight_decay=self.args['weight_decay'])

        # Training and validation loop        

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
                
                x = x.to(self.device)

                B, _, _ = x.shape
               
                aug1, aug2 = DataTransform(x)
                
                features_aug1 = self.net(aug1)
                features_aug2 = self.net(aug2)

                # normalize projection feature vectors
                features1 = F.normalize(features_aug1, dim=2)
                features2 = F.normalize(features_aug2, dim=2)

                temp_cont_loss1, temp_cont_lstm_feat1 = self.tc(features1, features2)
                temp_cont_loss2, temp_cont_lstm_feat2 = self.tc(features2, features1)

                # normalize projection feature vectors
                zis = temp_cont_lstm_feat1 
                zjs = temp_cont_lstm_feat2 

                lambda1 = 1
                lambda2 = 0.7
                nt_xent_criterion = NTXentLoss(self.device, B, 0.2,
                                            True)
                loss = (temp_cont_loss1 + temp_cont_loss2) * lambda1 +  nt_xent_criterion(zis, zjs) * lambda2
                
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
        out = self.net(x.to(self.device))

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
