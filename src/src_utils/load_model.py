from baselines.ts_tcc import TS_TCC
from baselines.tf_c import TF_C
from baselines.di_cot import Di_COT
from baselines.cost import CoST
from baselines.ts2vec import TS2Vec
from baselines.soft_ts2vec import Soft
from baselines.infots import InfoTS
from baselines.tnc import TNC
from baselines.simmtm import SimMTM
from baselines.rand_init import Rand_Init
from baselines.catt import CaTT
from baselines.supervised_b import Supervised_B
from baselines.supervisede2e import SupervisedE2E


def initialize_model(method, args, config, ds_args, device):

    if method == 'Di_COT':
        model = Di_COT(
            args,
            config,
            device=device
        )

    elif method == 'CaTT':
        model = CaTT(
            args,
            config,
            device=device
        )

    elif method == 'TS_TCC':
        model = TS_TCC(
            args,
            config,
            device=device
        )
    
    elif method == 'TF_C':
        model = TF_C(
            args,
            config,
            device=device
        )

    elif method == 'TS2Vec':
        model = TS2Vec(
            args,
            config,
            device=device
        )

    elif method == 'CoST':
        model = CoST(
            args,
            config,
            device=device
        )
    
    elif method == 'Soft':
        model = Soft(
            args,
            config,
            device=device
        )

    elif method == 'InfoTS':
        model = InfoTS(
            args,
            config,
            device=device
        )

    elif method == 'TNC':
        model = TNC(
            args,
            config,
            device=device
        )

    elif method == 'SimMTM':
        model = SimMTM(
            args,
            config,
            device=device
        )

    elif method == 'Rand_Init':
        model = Rand_Init(
            args,
            config,
            device=device
        )
    
    elif method == 'Supervised_B':
        model = Supervised_B(
            args,
            config,
            device=device
        )

    elif method == 'SupervisedE2E':
        model = SupervisedE2E(
            args,
            config,
            ds_args['num_labels'],
            device=device
        )

    else:
        raise ValueError(f"Unsupported BASELINE: {method}")

    return model