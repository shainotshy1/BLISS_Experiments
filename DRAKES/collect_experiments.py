from dataclasses import dataclass
from typing import Optional
import pandas as pd

@dataclass
class BLISSExperiment:
    name: str
    base_path: str
    model: Optional[str] = None
    dataset: Optional[str] = None
    align_type: Optional[str] = None
    align_n: Optional[int] = None
    oracle_mode: Optional[str] = None
    oracle_alpha: Optional[float] = None
    lasso_lambda: Optional[float] = None
    target_protein: Optional[str] = None

    def get_test_name(self) -> str:
        out_name = f"{self.model}" 
        if self.dataset == "single":
            out_name += f"_{self.target_protein}"
        else:
            out_name += f"_{self.dataset}"
        if self.oracle_mode:
            out_name += f"_{self.oracle_mode}"
        if self.oracle_mode == "balanced":
            out_name += f"_alpha={self.oracle_alpha}"
        if self.align_type and self.align_n:
            out_name += f"_{self.align_type}_N={self.align_n}"
        if self.align_type == "linear":
            out_name += f"_lambda={self.lasso_lambda}"
        return out_name
    
    def get_df(self) -> pd.DataFrame:
        data_path = f"{self.base_path}{self.get_test_name()}.csv"
        return pd.read_csv(data_path)

def collect_ddg_n10_experiments() -> list[BLISSExperiment]:
    bliss_dir = '/home/shai/BLISS_Experiments/DRAKES/'
    exp_dir = 'DRAKES/drakes_protein/fmif/eval_results/'
    dataset = 'test'
    experiments = []

    # Pretrained
    experiments.append(BLISSExperiment(
        name='Pretrained',
        base_path=bliss_dir + 'data_full/baseline/pretrained/distribution/',
        model='pretrained',
        dataset=dataset
    ))

    # Pretrained, ddg, LASSO, N=10, lambda=0.0005
    experiments.append(BLISSExperiment(
        name='Pretrained LASSO (N=10, λ=0.0005)',
        base_path=exp_dir + dataset + '/',
        model='pretrained',
        dataset=dataset,
        align_type='linear',
        align_n=10,
        oracle_mode='ddg',
        lasso_lambda=0.0005
    ))

    # Pretrained, ddg, BON, N=10
    experiments.append(BLISSExperiment(
        name='Pretrained BON (N=10)',
        base_path=exp_dir + dataset + '/',
        model='pretrained',
        dataset=dataset,
        align_type='bon',
        align_n=10,
        oracle_mode='ddg'
    ))

    # Pretrained, ddg, BEAM, N=10
    experiments.append(BLISSExperiment(
        name='Pretrained BEAM (N=10)',
        base_path=exp_dir + dataset + '/',
        model='pretrained',
        dataset=dataset,
        align_type='beam',
        align_n=10,
        oracle_mode='ddg'
    ))
    
    # DRAKES
    experiments.append(BLISSExperiment(
        name='DRAKES',
        base_path=bliss_dir + 'data_full/baseline/drakes/distribution/',
        model='drakes',
        dataset=dataset
    ))

    return experiments

def collect_ddg_n50_experiments() -> list[BLISSExperiment]:
    bliss_dir = '/home/shai/BLISS_Experiments/DRAKES/'
    exp_dir = 'DRAKES/drakes_protein/fmif/eval_results/'
    dataset = 'test'
    experiments = []

    # Pretrained
    experiments.append(BLISSExperiment(
        name='Pretrained',
        base_path=bliss_dir + 'data_full/baseline/pretrained/distribution/',
        model='pretrained',
        dataset=dataset
    ))

    # Pretrained, ddg, LASSO, N=50, lambda=0.0005
    experiments.append(BLISSExperiment(
        name='Pretrained LASSO (N=50, λ=0.0005)',
        base_path=exp_dir + dataset + '/',
        model='pretrained',
        dataset=dataset,
        align_type='linear',
        align_n=50,
        oracle_mode='ddg',
        lasso_lambda=0.005
    ))

    # Pretrained, ddg, SPECTRAL, N=50
    experiments.append(BLISSExperiment(
        name='Pretrained SPECTRAL (N=50)',
        base_path=exp_dir + dataset + '/',
        model='pretrained',
        dataset=dataset,
        align_type='spectral',
        align_n=50,
        oracle_mode='ddg'
    ))

    # Pretrained, ddg, BON, N=50
    experiments.append(BLISSExperiment(
        name='Pretrained BON (N=50)',
        base_path=exp_dir + dataset + '/',
        model='pretrained',
        dataset=dataset,
        align_type='bon',
        align_n=50,
        oracle_mode='ddg'
    ))

    # Pretrained, ddg, BEAM, N=50
    experiments.append(BLISSExperiment(
        name='Pretrained BEAM (N=50)',
        base_path=exp_dir + dataset + '/',
        model='pretrained',
        dataset=dataset,
        align_type='beam',
        align_n=50,
        oracle_mode='ddg'
    ))
    
    # DRAKES
    experiments.append(BLISSExperiment(
        name='DRAKES',
        base_path=bliss_dir + 'data_full/baseline/drakes/distribution/',
        model='drakes',
        dataset=dataset
    ))

    return experiments
