import torch
from tqdm import tqdm
from src.models.inceptiontime_pool import *
from src.src_utils.utils import cosine_warmup_scheduler
from src.models.resnet1D import *
from src.models.tcnencoder_pool import *
from src.models.dilatedcnn_pool import *
from pytorch_lightning.loggers import WandbLogger
import wandb
from models import TSEncoder
import os
import math
import time
from utils import name_with_datetime

class Di_COT:
    '''The Di_COT model'''
    
    def __init__(
        self,
        args,
        config,
        device='cuda',
    ):
        '''
          Initialize a Di_COT model.

        '''
        
        self.args = args
        self.config = config
        super().__init__()
        
        self.device = device

        self.net = InceptionTime(n_in_channels=args['feature_dim'], out_channels=args['out_features']).to(self.device)
        #self.net = ResNet1D(n_in_channels=args['feature_dim'], out_channels=args['out_features']).to(self.device)
        #self.net = DilatedCNN(input_dims=args['feature_dim'], in_channels = 64, channels = [args['out_features']], kernel_size = 3).to(self.device)


        self.n_iters = 0

        #self.projection_head = nn.Sequential(
        #                     nn.Linear(args['out_features'], args['out_features']//2),
        #                     nn.ReLU(),
        #                     nn.Linear(args['out_features']//2, args['out_features']//2)
        #                 ).to(self.device)
        

       
    
    def fit(self, train_dataset, ds_name, verbose=False):
        ''' Training the Di_COT model.
        
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
                drop_last = True,
            )
        
        # Wandb setup
        if self.config.WANDB:    
            proj_name = 'Di_COT' + ds_name + str(self.config.SEED)
            run_name = 'Di_COT'

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
#        optimizer = torch.optim.AdamW(list(self.net.parameters()) + list(self.projection_head.parameters()), lr=self.args['lr'], betas=(0.9, 0.99), weight_decay=self.args['weight_decay'])

        n_iters = self.args['iterations']
        pbar = tqdm(total=n_iters, desc="Training")
        epoch = 0
        num_training_steps = n_iters
        num_warmup_steps = int(0.1 * n_iters)

        scheduler = cosine_warmup_scheduler(optimizer, num_warmup_steps, num_training_steps)


        #n = self.args['n_split']
        p = self.args['p_overlap']
    

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

                batch, window, feat = x.shape

                # theoretical subwindow length
                #n = np.random.choice([5, 7, 10])
                n = np.random.randint(2, 11)
            
                L = window / (1 + (n - 1) * (1 - p))

                # force L to be even
                L = int(round(L / 2) * 2)

                # compute stride
                s = int(round(L * (1 - p)))

                n_subwindows = (window - L) // s + 1
                
                # Create strided view
                subwindows = x.as_strided(
                    size=(batch, n_subwindows, L, feat),
                    stride=(x.stride(0), s*x.stride(1), x.stride(1), x.stride(2))
                ).contiguous()

                batch, seq, window, feat = subwindows.shape

                combined_windows = subwindows.reshape(-1, L, feat)
                
                z = self.net(combined_windows)

                #z = self.projection_head(h)
                #z = F.normalize(z, dim=-1)

                zs = z.view(batch, seq, -1)

                temperature = 0.07
                similarities = torch.einsum('bkd,bjd->bkj', zs, zs) / temperature # (B, pred_K, future_len)

                positives = torch.arange(seq - 1, device=similarities.device)

                #g = torch.Generator(device=similarities.device)
                #g.manual_seed(42)

                #positives = positives[torch.randperm(len(positives), generator=g, device=positives.device)]

                positives = torch.cat([torch.zeros(1, device=positives.device, dtype=positives.dtype), positives])
                positives = positives.unsqueeze(0).repeat(batch, 1)

                # Compute cross-entropy: flatten batch and pred dims into one dimension
                similarities_flat = similarities.reshape(batch * seq, seq)   # (B*pred_K, future_len)
                positives_flat = positives.reshape(batch * seq)            # (B*pred_K,)

                loss = F.cross_entropy(similarities_flat, positives_flat)
                
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

