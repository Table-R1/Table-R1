set -x

MODEL_PATHS=(
    "Table-R1/Table-R1-SFT-7B"
    "Table-R1/Table-R1-Zero-7B"
    
    "Table-R1/Table-R1-SFT-8B"
    "Table-R1/Table-R1-Zero-8B"
)

N_PASSES=5
MAX_LENGTH=2048 # 18000 for reasoning model
TP_SIZE=1

home_dir='Table-R1'
TEST_FILE="$home_dir/data/table-r1-eval-dataset.parquet"

for MODEL_PATH in "${MODEL_PATHS[@]}"; do
    OUTPUT_DIR="$home_dir/results/$MODEL_PATH"
    mkdir -p ${OUTPUT_DIR}
    mkdir -p "$home_dir/eval_log/$MODEL_PATH"

    python3 -m verl.trainer.main_generation \
        trainer.nnodes=1 \
        trainer.n_gpus_per_node=4 \
        data.path=${TEST_FILE} \
        data.output_path=${OUTPUT_DIR}/table-r1-eval-${N_PASSES}.parquet \
        data.n_samples=${N_PASSES} \
        data.batch_size=2048 \
        model.path=${MODEL_PATH} \
        rollout.enforce_eager=False \
        rollout.free_cache_engine=False \
        rollout.max_num_batched_tokens=17000 \
        rollout.temperature=0.6 \
        rollout.prompt_length=14000 \
        rollout.response_length=${MAX_LENGTH} \
        rollout.top_k=-1 \
        rollout.top_p=0.95 \
        rollout.gpu_memory_utilization=0.9 \
        rollout.tensor_model_parallel_size=${TP_SIZE} $@ 2>&1 | tee >(split -b 5M -d --additional-suffix=.log - "$home_dir/eval_log/$MODEL_PATH/")
done
