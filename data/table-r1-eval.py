from datasets import load_dataset

dataset = load_dataset('Table-R1/Table-R1-Eval-Dataset', split='test')
dataset.to_parquet('table-r1-eval-dataset.parquet')
