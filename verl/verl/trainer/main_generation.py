# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Generate responses given a dataset of prompts
"""

import os

import hydra
import numpy as np
import ray
from tabulate import tabulate
import json
import csv
os.environ["NCCL_DEBUG"] = "WARN"
os.environ["TOKENIZERS_PARALLELISM"] = "true"
# os.environ['TORCH_COMPILE_DISABLE'] = '1'

from pprint import pprint

import pandas as pd
from omegaconf import OmegaConf

from verl import DataProto
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.single_controller.ray import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup
from verl.utils import hf_tokenizer
from verl.utils.fs import copy_to_local
from verl.utils.hdfs_io import makedirs
from verl.utils.model import compute_position_id_with_mask
from verl.workers.fsdp_workers import ActorRolloutRefWorker


@hydra.main(config_path="config", config_name="generation", version_base=None)
def main(config):
    run_generation(config)


def run_generation(config) -> None:
    if not ray.is_initialized():
        # this is for local ray cluster
        ray.init(runtime_env={"env_vars": {"TOKENIZERS_PARALLELISM": "true", "NCCL_DEBUG": "WARN"}})

    ray.get(main_task.remote(config))


@ray.remote(num_cpus=1)
def main_task(config):
    pprint(OmegaConf.to_container(config, resolve=True))  # resolve=True will eval symbol values
    OmegaConf.resolve(config)

    # Check if output file already exists
    if os.path.exists(config.data.output_path):
        print(f"Output file {config.data.output_path} already exists. Skipping generation and proceeding to evaluation.")
        try:
            dataset = pd.read_parquet(config.data.output_path)
        except Exception as e:
            print(f"Error reading parquet file: {e}")
    else:
        local_path = copy_to_local(config.model.path)
        trust_remote_code = config.data.get("trust_remote_code", False)
        tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)

        if config.rollout.temperature == 0.0:
            assert config.data.n_samples == 1, "When temperature=0, n_samples must be 1."
        assert config.data.n_samples >= 1, "n_samples should always >= 1"

        # read dataset. Note that the dataset should directly contain chat template format (e.g., a list of dictionary)
        dataset = pd.read_parquet(config.data.path)
        chat_lst = dataset[config.data.prompt_key].tolist()

        chat_lst = [chat.tolist() for chat in chat_lst]

        tokenizer.padding_side = "left"
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        ray_cls_with_init = RayClassWithInitArgs(cls=ray.remote(ActorRolloutRefWorker), config=config, role="rollout")
        resource_pool = RayResourcePool(process_on_nodes=[config.trainer.n_gpus_per_node] * config.trainer.nnodes)
        wg = RayWorkerGroup(resource_pool=resource_pool, ray_cls_with_init=ray_cls_with_init)
        wg.init_model()

        total_samples = len(dataset)
        config_batch_size = config.data.batch_size
        num_batch = -(-total_samples // config_batch_size)
        output_lst = [[] for _ in range(config.data.n_samples)]

        for batch_idx in range(num_batch):
            print(f"[{batch_idx + 1}/{num_batch}] Start to process.")
            batch_chat_lst = chat_lst[batch_idx * config_batch_size : (batch_idx + 1) * config_batch_size]
            inputs = tokenizer.apply_chat_template(
                batch_chat_lst,
                add_generation_prompt=True,
                padding=True,
                truncation=True,
                max_length=config.rollout.prompt_length,
                return_tensors="pt",
                return_dict=True,
                tokenize=True,
            )
            input_ids = inputs["input_ids"]
            attention_mask = inputs["attention_mask"]
            position_ids = compute_position_id_with_mask(attention_mask)
            batch_dict = {"input_ids": input_ids, "attention_mask": attention_mask, "position_ids": position_ids}

            data = DataProto.from_dict(batch_dict)
            data_padded, pad_size = pad_dataproto_to_divisor(data, wg.world_size)

            # START TO GENERATE FOR n_samples TIMES
            print(f"[{batch_idx + 1}/{num_batch}] Start to generate.")
            for n_sample in range(config.data.n_samples):
                output_padded = wg.generate_sequences(data_padded)
                output = unpad_dataproto(output_padded, pad_size=pad_size)

                output_texts = []
                for i in range(len(output)):
                    data_item = output[i]
                    prompt_length = data_item.batch["prompts"].shape[-1]
                    valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()
                    valid_response_ids = data_item.batch["responses"][:valid_response_length]
                    response_str = tokenizer.decode(valid_response_ids, skip_special_tokens=True)
                    output_texts.append(response_str)

                output_lst[n_sample].extend(output_texts)

        # convert output_lst from (n_samples, n_data) to (n_data, n_sampels)
        output_lst = np.array(output_lst, dtype=object)
        output_lst = np.transpose(output_lst, axes=(1, 0)).tolist()

        # add to the data frame
        dataset["responses"] = output_lst

        # write to a new parquet
        output_dir = os.path.dirname(config.data.output_path)
        makedirs(output_dir, exist_ok=True)
        dataset.to_parquet(config.data.output_path)

    output_dir = os.path.dirname(config.data.output_path)
    responses = dataset['responses']  # Using the generated responses
    data_sources = dataset['data_source']
    reward_model_data = dataset['reward_model']
    extra_info_data = dataset['extra_info']

    total = len(dataset)
    total_accurate_scores = []
    total_bleu_scores = []
    total_rouge_scores = []
    max_accurate_scores = []
    max_bleu_scores = []
    max_rouge_scores = []
    results = {}
    
    for i in range(total):
        response_lst = responses[i]
        reward_data = reward_model_data[i]
        id = extra_info_data[i]['id']
        task_type = extra_info_data[i]['task_type']
        reward_fn = select_reward_fn(task_type)
        ground_truth = reward_data['ground_truth']

        accurate_score_lst = []
        bleu_score_lst = []
        rouge_score_lst = []
        for r in response_lst:
            score_dict = reward_fn(r, ground_truth)
            accurate_score_lst.append(score_dict["accurate_score"])
            bleu_score_lst.append(score_dict["bleu_score"])
            rouge_score_lst.append(score_dict["rouge_score"])

        total_accurate_scores.append(accurate_score_lst)
        total_bleu_scores.append(bleu_score_lst)
        total_rouge_scores.append(rouge_score_lst)

        max_accurate_scores.append(np.max(accurate_score_lst))
        max_bleu_scores.append(np.max(bleu_score_lst))
        max_rouge_scores.append(np.max(rouge_score_lst))

        results[id] = {
            "accurate_score": accurate_score_lst,
            "bleu_score": bleu_score_lst,
            "rouge_score": rouge_score_lst,
        }

    n_samples = config.data.n_samples

    accurate_score_pass_at_1 = np.mean(total_accurate_scores)
    bleu_score_pass_at_1 = np.mean(total_bleu_scores)
    rouge_score_pass_at_1 = np.mean(total_rouge_scores)

    accurate_score_pass_at_n = np.mean(max_accurate_scores)
    bleu_score_pass_at_n = np.mean(max_bleu_scores)
    rouge_score_pass_at_n = np.mean(max_rouge_scores)

    csv_path = os.path.join(output_dir, f'pass_{n_samples}.csv')
    dataset_name = os.path.basename(config.data.path)
    row_data = {
        'model_path': config.model.path,
        'dataset': dataset_name,
        'accurate_score_pass@1': accurate_score_pass_at_1,
        'bleu_score_pass@1': bleu_score_pass_at_1,
        'rouge_score_pass@1': rouge_score_pass_at_1,
        'accurate_score_pass@n': accurate_score_pass_at_n,
        'bleu_score_pass@n': bleu_score_pass_at_n,
        'rouge_score_pass@n': rouge_score_pass_at_n,
    }

    file_exists = os.path.isfile(csv_path)
    with open(csv_path, mode='a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=row_data.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(row_data)

    table_data = [[k, v] for k, v in row_data.items()]
    print(tabulate(table_data, headers=['Metric', 'Value'], tablefmt='grid'))

    results_path = config.data.output_path.replace('.parquet', '.json')
    with open(results_path, 'w') as f:
        json.dump(results, f)

def select_reward_fn(task_type):
    if task_type == "TQA":
        from verl.utils.reward_score import tqa_eval
        return tqa_eval.compute_score
    elif task_type == "TFV":
        from verl.utils.reward_score import tfv_eval
        return tfv_eval.compute_score
    elif task_type == "FF-TQA":
        from verl.utils.reward_score import ff_tqa_eval
        return ff_tqa_eval.compute_score
    else:
        raise ValueError(f"Unsupported task type: {task_type}")

if __name__ == "__main__":
    main()
