from torch.nn import GRU, Linear, CrossEntropyLoss
import wandb
import torch
from models.contrastive import LS_HATCL_LOSS, HATCL_LOSS
from tqdm import tqdm
# from src.models.attention_model import *
from src.models.inceptiontime import *
from pytorch_lightning.loggers import WandbLogger
import wandb
import numpy as np
from src.src_utils.utils import cosine_warmup_scheduler
import time
from utils import name_with_datetime
import os


class SupervisedPretrainModel(nn.Module):
    def __init__(self, feature_extractor, feature_dim, num_classes):
        super().__init__()
        self.feature_extractor = feature_extractor
        self.classifier = nn.Linear(feature_dim, num_classes)

    def forward(self, x):
        features = self.feature_extractor(x)  # [B, feature_dim]
        features = F.max_pool1d(
                features.transpose(1, 2),
                kernel_size = features.size(1),
            ).transpose(1, 2).squeeze()
        logits = self.classifier(features)    # [B, num_classes]
        return logits
    
class SupervisedE2E:

    '''The SupervisedE2E model'''
    def __init__(
        self,
        args,
        config,
        num_labels,
        device='cuda',
    ):
        '''
          Initialize a SupervisedE2E model.

        '''
        
        self.args = args
        self.config = config
        super().__init__()
        
        self.device = device
        # self.net = FeatureProjector(input_size=args['feature_dim'], output_size=args['out_features']).to(self.device)
        self.net = InceptionTime(n_in_channels=args['feature_dim'], out_channels=args['out_features']).to(self.device)

        self.wrapped_model = None
        
        self.n_iters = 0
        
        self.wrapped_model = SupervisedPretrainModel(
                        feature_extractor=self.net,
                        feature_dim=args['out_features'],
                        num_classes=num_labels
                    ).to(self.device)
       
    
    def fit(self, train_dataset, ds_name, num_labels, args, verbose=False):
        ''' Training the SupervisedE2E model.
        
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
            proj_name = 'SupervisedE2E' + ds_name + str(self.config.SEED)
            run_name = 'SupervisedE2E'

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

        epochs = args['supervised_epochs']
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(self.wrapped_model.parameters(), lr=float(args['lr']), weight_decay=float(args['weight_decay']))
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

        run_dir = f'results/{ds_name}/seed_{self.config.SEED}/{name_with_datetime(self.__class__.__name__)}'
        os.makedirs(run_dir, exist_ok=True)
        start_time = time.time()

        for epoch in tqdm(range(epochs)):
            self.wrapped_model.train()
            running_loss = 0.0
            for x, y in train_loader:
                
                x, y = x.to(self.device).float(), y.to(self.device).long()
                
                optimizer.zero_grad()
                
                logits = self.wrapped_model(x)

                loss = criterion(logits, y)
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
            
            scheduler.step()

            if verbose:
                print(f"Epoch {epoch}, Train Loss: {running_loss:.4f}")

            # Log training loss to Wandb
            if self.config.WANDB:
                wandb.log({'Train Loss': running_loss, 'Epoch': epoch})

        # Save model
        model_path = os.path.join(run_dir, f'model.pt')
        torch.save(self.wrapped_model.state_dict(), model_path)

        total_time = time.time() - start_time

        # Save training time
        time_file = os.path.join(run_dir, 'time.txt')
        with open(time_file, 'w') as f:
            f.write(str(total_time))

        try:   
            return running_loss
        except:
            return 0

    def encode(self, x):
        self.net.eval()
        out = self.wrapped_model(x.to(self.device))

        return out


    def save(self, fn):
        ''' Save the model to a file.
        
        Args:
            fn (str): filename.
        '''
        torch.save(self.wrapped_model.state_dict(), fn)
    
    def load(self, fn):
        ''' Load the model from a file.
        
        Args:
            fn (str): filename.
        '''
        state_dict = torch.load(fn, map_location=self.device)
        self.wrapped_model.load_state_dict(state_dict)
