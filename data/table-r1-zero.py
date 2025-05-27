from datasets import load_dataset

dataset = load_dataset('Table-R1/Table-R1-Zero-Dataset')
train_dataset = dataset['train']
test_dataset = dataset['test']

train_dataset.to_parquet('table-r1-zero-dataset_train.parquet')
test_dataset.to_parquet('table-r1-zero-dataset_test.parquet')
