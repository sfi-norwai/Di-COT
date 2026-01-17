import torch
from utils import take_per_row
from tqdm import tqdm
import torch
from models.contrastive import LS_HATCL_LOSS, HATCL_LOSS
from pytorch_lightning.loggers import WandbLogger
import wandb
from models.soft_losses import *
# from tslearn.metrics import dtw, dtw_path,gak
from sklearn.preprocessing import MinMaxScaler
from src.models.ts2vecencoder import *
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from src.src_utils.utils import cosine_warmup_scheduler
import time
from utils import name_with_datetime
import os
from tslearn.metrics import dtw, dtw_path,gak
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
from torch.utils.data import DataLoader, Dataset, Sampler
import math
from torch.utils.data import Subset
from dtaidistance import dtw_ndim

def get_COS(MTS_tr):
    # Ensure input is torch.Tensor
    if not isinstance(MTS_tr, torch.Tensor):
        MTS_tr = torch.tensor(MTS_tr, dtype=torch.float32)
    # Flatten along time and channels
    MTS_tr = MTS_tr.view(MTS_tr.shape[0], -1)
    cos_sim_matrix = -cosine_similarity(MTS_tr)
    return cos_sim_matrix

def get_EUC(MTS_tr):
    # Convert to tensor if needed
    if isinstance(MTS_tr, torch.Tensor):
        MTS_tr = MTS_tr.cpu().numpy()
    elif isinstance(MTS_tr, list):
        MTS_tr = np.array(MTS_tr, dtype=np.float32)

    # Flatten along time and channels
    N = MTS_tr.shape[0]
    MTS_flat = MTS_tr.reshape(N, -1)

    # Compute Euclidean distance
    dist_matrix = euclidean_distances(MTS_flat)
    return dist_matrix.astype(np.float32)

def get_DTW(UTS_tr):
    N = len(UTS_tr)
    dist_mat = np.zeros((N,N))
    for i in tqdm(range(N)):
        for j in range(N):
            if i>j:
                dist = dtw(UTS_tr[i].reshape(-1,1), UTS_tr[j].reshape(-1,1))
                dist_mat[i,j] = dist
                dist_mat[j,i] = dist
            elif i==j:
                dist_mat[i,j] = 0
            else :  
                pass
    return dist_mat

def get_MDTW(MTS_tr):
    N = MTS_tr.shape[0]

    dist_mat = np.zeros((N,N))
    for i in tqdm(range(N)):
        for j in range(N):
            if i>j:
                mdtw_dist = dtw(MTS_tr[i], MTS_tr[j])
                dist_mat[i,j] = mdtw_dist
                dist_mat[j,i] = mdtw_dist
            elif i==j:
                dist_mat[i,j] = 0
            else :
                pass
    return dist_mat

def save_sim_mat(X_tr, min_=0, max_=1, multivariate=True, type_='DTW'):
    """
    X_tr: numpy array (N, T, C) if multivariate True else (N, T)
    returns normalized similarity matrix (N x N) with diagonal filled minimally
    """
    N = len(X_tr)
    if multivariate:
        assert type_ == 'DTW'
        dist_mat = get_MDTW(X_tr)
    else:
        if type_ == 'DTW':
            dist_mat = get_DTW(X_tr)
        elif type_ == 'COS':
            dist_mat = get_COS(X_tr)
        elif type_ == 'EUC':
            dist_mat = get_EUC(X_tr)
        else:
            raise ValueError(type_)

    # (1) distance matrix
    diag_indices = np.diag_indices(N)
    mask = np.ones(dist_mat.shape, dtype=bool)
    mask[diag_indices] = False
    temp = dist_mat[mask].reshape(N, N - 1)
    dist_mat[diag_indices] = temp.min(axis=1)

    # (2) normalize distance matrix
    scaler = MinMaxScaler(feature_range=(min_, max_))
    dist_mat_norm = scaler.fit_transform(dist_mat)

    # (3) convert to similarity
    sim_mat = 1.0 - dist_mat_norm
    return sim_mat.astype(np.float32)

# ---------------------------
# Custom batch sampler
# ---------------------------

def compute_full_soft_labels(dataset, multivariate=True, type_='DTW', radius=1):
    """Compute the full NxN soft label matrix once before training"""
    X = []
    for i in range(len(dataset)):
        item = dataset[i][0]
        if isinstance(item, torch.Tensor):
            arr = item.cpu().numpy()
        else:
            arr = np.asarray(item)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        X.append(arr)
    X = np.array(X, dtype=float)
    return save_sim_mat(X, multivariate=multivariate, type_=type_)

class MTS_Dataset(Dataset):
    def __init__(self, data):
        self.data = data  # list or np.array of (T, C)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], idx  # return index for precomputed labels
    
class IndexBatchSampler(Sampler):
    def __init__(self, dataset_len, batch_size, drop_last=False):
        self.dataset_len = dataset_len
        self.batch_size = batch_size
        self.drop_last = drop_last

    def __iter__(self):
        indices = torch.randperm(self.dataset_len).tolist()
        for start in range(0, self.dataset_len, self.batch_size):
            batch_idx = indices[start:start + self.batch_size]
            if len(batch_idx) < self.batch_size and self.drop_last:
                break
            yield batch_idx

    def __len__(self):
        if self.drop_last:
            return self.dataset_len // self.batch_size
        else:
            return math.ceil(self.dataset_len / self.batch_size)
        
def soft_collate(batch):
    """
    batch: list of (data, index)
    Returns:
        x: tensor (B, T_max, C)
        idxs: tensor (B,)
    """
    xs = []
    idxs = []
    for data, idx in batch:
        if isinstance(data, torch.Tensor):
            data = data.cpu().numpy()
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        xs.append(data)
        idxs.append(idx)

    max_len = max(x.shape[0] for x in xs)
    C = xs[0].shape[1]
    padded = []
    for x in xs:
        pad_len = max_len - x.shape[0]
        if pad_len > 0:
            x = np.pad(x, ((0, pad_len), (0, 0)))
        padded.append(x)

    x = torch.tensor(np.stack(padded), dtype=torch.float32)
    idxs = torch.tensor(idxs, dtype=torch.long)
    return x, idxs

class Soft:
    '''The Soft model'''
    
    def __init__(
        self,
        args,
        config,
        device='cuda',
    ):
        '''
          Initialize a Soft model.

        '''
        
        self.args = args
        self.config = config
        super().__init__()
        
        self.device = device
        self.temporal_unit = 0
        self.max_train_length = None
        self.net = TSEncoder(input_dims=args['feature_dim'], output_dims=args['out_features']).to(self.device)
        self.n_iters = 0
       
    
    def fit(self, train_dataset, ds_name, verbose=False):
        ''' Training the Soft model.
        
        Args:
            train_data (numpy.ndarray): The training data. It should have a shape of (n_instance, n_timestamps, n_features). All missing data should be set to NaN.
            n_epochs (Union[int, NoneType]): The number of epochs. When this reaches, the training stops.
            n_iters (Union[int, NoneType]): The number of iterations. When this reaches, the training stops.
        Returns:
            loss_log: a list containing the training losses on each epoch.
        '''
 
        batch_sampler = IndexBatchSampler(len(train_dataset), batch_size=self.args['batch_size'], drop_last=True)

        train_loader = torch.utils.data.DataLoader(
                dataset=train_dataset,
                batch_sampler=batch_sampler,
                collate_fn=soft_collate
            )
        
        # Wandb setup
        if self.config.WANDB:    
            proj_name = 'Dynamic_CL' + ds_name + str(self.config.SEED)
            run_name = 'Soft'

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

        optimizer = torch.optim.AdamW(self.net.parameters(), lr=self.args['lr'], betas=(0.9, 0.99), weight_decay=self.args['weight_decay'])

        lambda_ = 0.5
        tau_temp = 2
        temporal_unit = 0
        soft_instance = True
        soft_temporal = False
        

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

        print("Precomputing soft labels...")
        self.soft_label_matrix = compute_full_soft_labels(train_dataset, multivariate=False, type_='COS')
        
        while True:

            # Training phase
            self.net.train()  # Set the model to training mode
            train_running_loss = 0.0
            n_epoch_iters = 0

            for x, idxs in train_loader:

                interrupted = False
                if n_iters is not None and self.n_iters >= n_iters:
                    interrupted = True
                    break

                x = x.to(self.device)
                soft_labels_batch = self.soft_label_matrix[np.ix_(idxs, idxs)]
                soft_labels_batch = torch.tensor(soft_labels_batch, dtype=torch.float32, device=self.device)
                #soft_labels_batch = torch.from_numpy(soft_labels_batch).float().to(self.device)

                if self.max_train_length is not None and x.size(1) > self.max_train_length:
                    window_offset = np.random.randint(x.size(1) - self.max_train_length + 1)
                    x = x[:, window_offset : window_offset + self.max_train_length]
                x = x.to(self.device)
                
                ts_l = x.size(1)
                crop_l = np.random.randint(low=2 ** (self.temporal_unit + 1), high=ts_l+1)
                crop_left = np.random.randint(ts_l - crop_l + 1)
                crop_right = crop_left + crop_l
                crop_eleft = np.random.randint(crop_left + 1)
                crop_eright = np.random.randint(low=crop_right, high=ts_l + 1)
                crop_offset = np.random.randint(low=-crop_eleft, high=ts_l - crop_eright + 1, size=x.size(0))
                
                optimizer.zero_grad()
                
                out1 = self.net(take_per_row(x, crop_offset + crop_eleft, crop_right - crop_eleft))
                out1 = out1[:, -crop_l:]
                
                out2 = self.net(take_per_row(x, crop_offset + crop_left, crop_eright - crop_left))
                out2 = out2[:, :crop_l]
                
                loss = hier_CL_soft(
                    out1,
                    out2,
                    soft_labels_batch,
                    lambda_= lambda_,
                    tau_temp = tau_temp,
                    temporal_unit = temporal_unit,
                    soft_temporal = soft_temporal, 
                    soft_instance = soft_instance
                )
                
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

    def encode(self, x, mask=None):
        self.net.eval()
        out = self.net(x.to(self.device, non_blocking=True), mask)

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
