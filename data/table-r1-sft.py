from datasets import load_dataset

dataset = load_dataset('Table-R1/Table-R1-SFT-Dataset-filtered', split='train')
dataset.to_parquet('table-r1-sft-dataset-filtered.parquet')
