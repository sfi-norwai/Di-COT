import torch
from models.contrastive import LS_HATCL_LOSS, HATCL_LOSS
from tqdm import tqdm
from src.models.attention_model import *
from src.models.inceptiontime_pool import *

from pytorch_lightning.loggers import WandbLogger
import wandb
from models import TSEncoder
from statsmodels.tsa.stattools import adfuller
import numpy as np
import math
from src.src_utils.utils import cosine_warmup_scheduler
import time
from utils import name_with_datetime
import os
from src.loader.dataloader import TNCDatasetFromDataset

class Discriminator(torch.nn.Module):
    def __init__(self, input_size, device):
        super(Discriminator, self).__init__()
        self.device = device
        self.input_size = input_size

        self.model = torch.nn.Sequential(torch.nn.Linear(2*self.input_size, 4*self.input_size),
                                         torch.nn.ReLU(inplace=True),
                                         torch.nn.Dropout(0.5),
                                         torch.nn.Linear(4*self.input_size, 1))

        torch.nn.init.xavier_uniform_(self.model[0].weight)
        torch.nn.init.xavier_uniform_(self.model[3].weight)

    def forward(self, x, x_tild):
        """
        Predict the probability of the two inputs belonging to the same neighbourhood.
        """
        x_all = torch.cat([x, x_tild], -1)
        p = self.model(x_all)
        return p.view((-1,))

class TNC:
    '''The TNC model'''
    
    def __init__(
        self,
        args,
        config,
        device='cuda',
    ):
        '''
          Initialize a TNC model.

        '''
        
        self.args = args
        self.config = config
        super().__init__()
        
        self.device = device

        # self.net = FeatureProjector(input_size=args['feature_dim'], output_size=args['out_features']).to(self.device)
        self.net = InceptionTime(n_in_channels=args['feature_dim'], out_channels=args['out_features']).to(self.device)
        self.disc_model = Discriminator(self.args['out_features'], self.device).to(self.device)

        self.n_iters = 0

    def fit(self, train_dataset, ds_name, verbose=False):
        ''' Training the TNC model.
        
        Args:
            train_data (numpy.ndarray): The training data. It should have a shape of (n_instance, n_timestamps, n_features). All missing data should be set to NaN.
            n_epochs (Union[int, NoneType]): The number of epochs. When this reaches, the training stops.
            n_iters (Union[int, NoneType]): The number of iterations. When this reaches, the training stops. If both n_epochs and n_iters are not specified, a default setting would be used that sets n_iters to 200 for a dataset with size <= 100000, 600 otherwise.
            verbose (bool): Whether to print the training loss after each epoch.
            
        Returns:
            loss_log: a list containing the training losses on each epoch.
        '''
        
        
        
        tcn_dataset = TNCDatasetFromDataset(
            dataset=train_dataset,
            mc_sample_size=20,
            window_size=self.args['sequence_sample']//10,
            augmentation=1
        )

        train_loader = torch.utils.data.DataLoader(
                dataset=tcn_dataset,
                batch_size= self.args['batch_size'],
                shuffle = True,
                num_workers=self.config.NUM_WORKERS,
                drop_last = True,
            )

        # Wandb setup
        if self.config.WANDB:    
            proj_name = 'Dynamic_CL' + ds_name + str(self.config.SEED)
            run_name = 'TNC'

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
        
        

        # Define loss function and optimizer
        self.args['lr'] = float(self.args['lr'])
        self.args['weight_decay'] = float(self.args['weight_decay'])
        
        optimizer = torch.optim.AdamW([
            {'params': self.net.parameters(), 'lr': self.args['lr']},
            {'params': self.disc_model.parameters(), 'lr': self.args['lr']}
        ],
        betas=(0.9, 0.999),
        weight_decay=self.args['weight_decay'])

        loss_fn = torch.nn.BCEWithLogitsLoss()

        # Training and validation loop
        w = 0.1

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

            for x_t, x_p, x_n, _ in train_loader:

                interrupted = False
                if n_iters is not None and self.n_iters >= n_iters:
                    interrupted = True
                    break

                mc_sample = x_p.shape[1]
                batch_size, f_size, len_size = x_t.shape
                x_p = x_p.reshape((-1, f_size, len_size))
                x_n = x_n.reshape((-1, f_size, len_size))
                x_t = np.repeat(x_t, mc_sample, axis=0)
                neighbors = torch.ones((len(x_p))).to(self.device)
                non_neighbors = torch.zeros((len(x_n))).to(self.device)

                x_t = x_t.transpose(2,1).to(self.device)
                x_p, x_n = x_p.transpose(2,1).to(self.device), x_n.transpose(2, 1).to(self.device)
                
                z_t = self.net(x_t)
                z_p = self.net(x_p)
                z_n = self.net(x_n)

                d_p = self.disc_model(z_t, z_p)
                d_n = self.disc_model(z_t, z_n)

                p_loss = loss_fn(d_p, neighbors)
                n_loss = loss_fn(d_n, non_neighbors)
                n_loss_u = loss_fn(d_n, neighbors)
                loss = (p_loss + w*n_loss_u + (1-w)*n_loss)/2
                
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
    
    def find_neighours(self, x, t, window_size):
    
        T = self.args['sequence_sample']
        mc_sample_size = self.args['tnc_window']
        adf = True
        
        if adf:
            
            gap = window_size
            corr = []
            for w_t in range(window_size,4*window_size, gap):
                try:
                    p_val = 0
                    for f in range(x.shape[-2]):

                        p = adfuller(np.array(x[f, max(0,t - w_t):min(x.shape[-1], t + w_t)].reshape(-1, )))[1]
                        p_val += 0.01 if math.isnan(p) else p
                    corr.append(p_val/x.shape[-2])
                except:
                    corr.append(0.6)
            epsilon = len(corr) if len(np.where(np.array(corr) >= 0.01)[0])==0 else (np.where(np.array(corr) >= 0.01)[0][0] + 1)
            delta = 5*epsilon*window_size

        ## Random from a Gaussian
        t_p = [int(t+np.random.randn()*epsilon*window_size) for _ in range(mc_sample_size)]
        t_p = [max(window_size//2+1,min(t_pp,T-window_size//2)) for t_pp in t_p]
        x_p = torch.stack([x[:, t_ind-window_size//2:t_ind+window_size//2] for t_ind in t_p])
        
        return x_p, delta

    def find_non_neighours(self, x, t, delta, window_size):
        T = self.args['sequence_sample']
        mc_sample_size = self.args['tnc_window']
        adf = True
        
        if t>T/2:
            t_n = np.random.randint(window_size//2, max((t - delta + 1), window_size//2+1), mc_sample_size)
        else:
            t_n = np.random.randint(min((t + delta), (T - window_size-1)), (T - window_size//2), mc_sample_size)
        x_n = torch.stack([x[:, t_ind-window_size//2:t_ind+window_size//2] for t_ind in t_n])

        if len(x_n)==0:
            rand_t = np.random.randint(0,window_size//5)
            if t > T / 2:
                x_n = x[:,rand_t:rand_t+window_size].unsqueeze(0)
            else:
                x_n = x[:, T - rand_t - window_size:T - rand_t].unsqueeze(0)
        return x_n

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
