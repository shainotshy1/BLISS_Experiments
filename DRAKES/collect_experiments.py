from dataclasses import dataclass
from typing import Optional
import pandas as pd
import os

from utils import get_eval_stats, analyze_protein_gen_helper, analyze_protein_gen_helper_violin
import seaborn as sn
import importlib

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
    
    def get_stats(self) -> dict:
        df = self.get_df()
        return get_eval_stats(df)

# Hardcoded labels for graphing
experiment_names = {
    'pretrained_test_ddg_linear_N=10_lambda=0.0005': 'Pretrained LASSO (N=10, λ=0.0005)',
    'pretrained_test_ddg_linear_N=50_lambda=0.005': 'Pretrained LASSO (N=50, λ=0.0005)',
    'pretrained_test_ddg_spectral_N=50': 'Pretrained SPECTRAL (N=50)',
    'pretrained_test_ddg_bon_N=10': 'Pretrained BON (N=10)',
    'pretrained_test_ddg_beam_N=10': 'Pretrained BEAM (N=10)',
    'pretrained_test_ddg_bon_N=50': 'Pretrained BON (N=50)',
    'pretrained_test_ddg_bon_N=100': 'Pretrained BON (N=100)',
    'pretrained_test_ddg_beam_N=50': 'Pretrained BEAM (N=50)',
    'pretrained_test_ddg_beam_N=100': 'Pretrained BEAM (N=100)',
    'pretrained_test_protgpt_bon_N=10': 'Pretrained BON (N=10)',
    'pretrained_test_protgpt_bon_N=50': 'Pretrained BON (N=50)',
    'pretrained_test_protgpt_beam_N=10': 'Pretrained BEAM (N=10)',
    'drakes_test': 'DRAKES',
    'pretrained_test': 'Pretrained'
}


# Based off order of parsing: model, [target_protein], dataset, oracle_mode, [oracle_alpha], align_type, align_n, [lasso_lambda]
def collect_experiments(n, oracle, dataset, model) -> list[BLISSExperiment]:
    bliss_dir = '/home/shai/BLISS_Experiments/DRAKES/'
    exp_dir = 'DRAKES/drakes_protein/fmif/eval_results/'
    base_path = bliss_dir + exp_dir + dataset + '/'
    assert dataset in ['test', 'validation', 'train'] # TODO: support 'single' parsing
    assert model in ['all', 'pretrained', 'drakes']
    assert oracle in ['ddg', 'protgpt', 'balanced'] # TODO: support scrmsd
    assert type(n) == int and n > 0

    # Collect experiment names 
    target_dir = bliss_dir + exp_dir + dataset + '/'
    all_experiments_fn = [f for f in os.listdir(target_dir) if f.endswith('.csv')]
    valid_experiments = []
    for f in all_experiments_fn:
        f = f[:-4]  # Remove .csv
        exp_name = experiment_names.get(f, f)
        exp = BLISSExperiment(exp_name, base_path) # type: ignore
        components = f.split('_')

        if model != 'all' and components[0] != model: continue
        exp.model = components[0]

        idx_offset = 0
        if exp.model == 'single':
            exp.target_protein = components[1]
            idx_offset += 1

        if components[1 + idx_offset] != dataset: continue
        exp.dataset = dataset

        if len(components) > 3: # Process inference alignment experiments
            exp.oracle_mode = components[2 + idx_offset]
            if exp.oracle_mode != oracle: continue
            if exp.oracle_mode == 'balanced':
                exp.oracle_alpha = float(components[3 + idx_offset][6:]) # Removing prefix 'alpha='
                idx_offset += 1
            
            exp.align_type = components[3 + idx_offset]
            exp.align_n = int(components[4 + idx_offset][2:]) # Removing prefix 'N='
            if exp.align_n != n: continue

            if exp.align_type == 'linear':
                exp.lasso_lambda = float(components[5 + idx_offset][7:]) # Removing prefix 'lambda='
                idx_offset += 1

        valid_experiments.append(exp)
        print(exp.name, exp.get_stats())
        
    return valid_experiments

def display_experiments(n, oracle, dataset='test', model='all', target_protein=None):
    experiments = collect_experiments(n, oracle, dataset, model)
    data = [exp.get_df() for exp in experiments]
    labels = [exp.name for exp in experiments]
    colors = sn.color_palette("Set2", len(experiments))
    protein_output = target_protein + " " if target_protein is not None else ""
    
    if oracle == 'ddg':
        oracle_name = 'ΔΔG'
    elif oracle == 'protgpt':
        oracle_name = 'Log Likelihood'
    elif oracle == 'balanced':
        oracle_name = 'Balanced'
    else:
        raise ValueError(f"Unknown oracle: {oracle}")
    analyze_protein_gen_helper_violin(target_protein, data, labels, colors, 'ddg_eval', y_label='Predicted ΔΔG', legend_pos='right', title=f'{protein_output}ΔΔG Evaluation: {oracle_name} Alignment (N={n})')
    analyze_protein_gen_helper_violin(target_protein, data, labels, colors, 'loglikelihood', y_label='Log Likelihood', legend_pos='right', title=f'{protein_output}Log Likelihood Evaluation: {oracle_name} Alignment (N={n})')
