# Table-R1: Inference-Time Scaling for Table Reasoning

<p align="center">
    🤗 <a href="https://huggingface.co/Table-R1">Hugging Face</a>&nbsp&nbsp | &nbsp&nbsp📄 <a href="https://arxiv.org/pdf/2505.23621">arXiv</a>
</p>

## 📖 Abstract
**Table-R1** introduces the first systematic study of inference-time scaling for table reasoning tasks. We develop two post-training strategies: **distillation from frontier model reasoning traces** and **reinforcement learning with verifiable rewards (RLVR)**. Our 7B-parameter **Table-R1-Zero** model matches or surpasses GPT-4.1 and DeepSeek-R1 performance while exhibiting strong generalization to out-of-domain datasets.

<div align="center">
<img src="./assets/overview.jpg" width="80%"/>
<p><em>Overview of the Table-R1.</em></p>
</div>

## 🚀 Quick Start
### Installation
```bash
git clone https://github.com/Table-R1/Table-R1.git

# Install verl framework
cd Table-R1/verl
pip install -e .

cd ..
pip install -r requirements.txt
```

## 🛠️ Training
### Table-R1-SFT Training
```bash
# Prepare SFT dataset
python data/table-r1-sft.py

# Run SFT training
bash script/table-r1-sft.sh
```

### Table-R1-Zero Training
```bash
# Prepare RLVR dataset  
python data/table-r1-zero.py

# Run RLVR training
bash script/table-r1-zero.sh
```

## 📈 Evaluation
```bash
# Prepare evaluation dataset
python data/table-r1-eval.py

# Run evaluation
bash script/table-r1-eval.sh
```

## Acknowledgements
- All models are trained using the excellent [verl](https://github.com/volcengine/verl) framework

## Citation
If you find Table-R1 useful in your research, please cite our paper:

```bibtex
@article{yang2025tabler1,
  title={Table-R1: Inference-Time Scaling for Table Reasoning},
  author={Yang, Zheyuan and Chen, Lyuhao and Cohan, Arman and Zhao, Yilun},
  journal={arXiv preprint arXiv:2505.23621},
  year={2025}
}
}

```
