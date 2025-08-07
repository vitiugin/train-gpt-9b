#!/bin/bash
#SBATCH --job-name=train-gpt
#SBATCH --cpus-per-task=7
#SBATCH --ntasks-per-node=8
#SBATCH --nodes=16
#SBATCH --mem=480G
#SBATCH --partition=standard-g
#SBATCH --time=40:00:00
#SBATCH --exclusive
#SBATCH --gpus-per-node=mi250:8
#SBATCH --account=project_462000353
#SBATCH --output logs/%j.out
#SBATCH --error logs/%j.err

# This is a slurm script for training generative models on LUMI using
# Megatron-LM pretrain_gpt.py. This script defines defaults for
# training a FineWeb-like model (approx. 1.7B parameters) and is
# intended to be reasonably easily modified for other model sizes
# by editing the variables defined in the "MODEL AND PRETRAINING
# CONFIGURATION" section below.
#
# Note that while the script defines default arguments for sbatch
# in the #SBATCH comments above, you can override any of these on the
# command line. For example, to run on 16 nodes:
#
#    sbatch --nodes 16 ./train.sh [...]

######################################################################
#
# ENVIRONMENT SETUP AND GENERAL CONFIGURATION
#
# This section of the script sets up the execution environment (logs,
# container, etc.) and configuration that is independent of the model
# or pretraining setup. It should generally not be necessary to edit
# this section, and you may wish to double-check that you understand
# what you are doing before you do.
#
######################################################################

# If this script is run without sbatch, invoke with sbatch here. This
# also gives us an opportunity to make sure logs/ exists. (If the
# directory where --output and/or --error are directed to doesn't
# exist, the run will fail silently.)
if [ -z $SLURM_JOB_ID ]; then
    mkdir -p logs
    sbatch "$0" "$@"
    exit
fi

# Bash "strict mode"
# (see http://redsymbol.net/articles/unofficial-bash-strict-mode/)
set -euo pipefail

# When slurm reschedules a job that ended on node failure, it will run
# with the same job ID, clobbering the original logs. Rename the logs
# and include timestamp to avoid this.
timestamp=$(date +"%Y-%m-%d_%H-%M-%S")
logfile_basename="${SLURM_JOB_NAME}-${SLURM_JOBID}-${timestamp}"
mv -f "logs/${SLURM_JOBID}.out" "logs/${logfile_basename}.out"
mv -f "logs/${SLURM_JOBID}.err" "logs/${logfile_basename}.err"

# Check if this is a restared run and if so, print the failure
# events/reasons for failed nodes. (This relies on "logs/latest.err"
# pointing to the error log of the failed run.)
if [[ -v SLURM_RESTART_COUNT ]]; then
    failed_node=$(grep 'Node failure' logs/latest.err | awk '{print $NF}')
    if [[ -z ${failed_node:+x} ]]; then
        echo "RUN RESTARTED but no node failure logged"
    else
        failed_node="${failed_node//$'\n'/ }"
        echo "RUN RESTARTED AFTER FAILURE OF NODE(s) $failed_node. Reason:"
        sacctmgr show event where node="$failed_node" format="NodeName,TimeStart,TimeEnd,State,Reason%100"
    fi
fi

# Symlink logs/latest.out and logs/latest.err for convenience and to
# support the above check.
ln -sf "${logfile_basename}.out" "logs/latest.out"
ln -sf "${logfile_basename}.err" "logs/latest.err"

# No modules are needed with the container we are using.
module purge

# Dedicated LUMI container with most of what we need for LLM training
CONTAINER=/appl/local/containers/sif-images/lumi-pytorch-rocm-6.2.4-python-3.12-pytorch-v2.6.0.sif

# Directories to map into container
BIND_DIRS="/pfs,/scratch,/projappl,/project,/flash,/appl,/usr/lib64/libjansson.so.4,/usr/lib64/libcxi.so.1,/opt/cray,/var/spool/slurmd" ### Ville's script: used 'SINGULARITY_BIND' instead of 'BIND_DIRS'. Added '/boot' to the bind directories. Removed: '/usr/lib64/libjansson.so.4', '/usr/lib64/libcxi.so.1'

# Avoid conflicts with $HOME/.local
export PYTHONUSERBASE=""

# Compilers in the container
export CC=gcc-12
export CXX=g++-12

# Mask to bind tasks to CPUs for one thread per core
c="fe"
BIND_MASK="0x${c}000000000000,0x${c}00000000000000,0x${c}0000,0x${c}000000,0x${c},0x${c}00,0x${c}00000000,0x${c}0000000000"

# PATHS
BASE_DIR="$SLURM_SUBMIT_DIR"
OUTPUT_DIR="$BASE_DIR/output"
CHECKPOINT_PATH="$OUTPUT_DIR/checkpoints"
TENSORBOARD_DIR="$OUTPUT_DIR/tensorboard/$SLURM_JOB_NAME-$SLURM_JOBID"
WANDB_DIR="$OUTPUT_DIR/wandb"

mkdir -p "$CHECKPOINT_PATH"    # This needs to exist

# WEIGHTS & BIASES CONFIG
WANDB_PROJECT="openeurollm"
WANDB_EXP_NAME="test"

# Script that is used to launch on GPU nodes
LAUNCH_SCRIPT="$BASE_DIR/launch.sh"

# Needed for sequence paralellism
# (see https://github.com/NVIDIA/Megatron-LM/issues/533)
export CUDA_DEVICE_MAX_CONNECTIONS=1

# DISTRIBUTED ARGS
# These are used by torch.distributed to allow the different processes
# to find each other. Note that RANK and LOCAL_RANK are also expected,
# but can only be set in the launcher script as the values are
# specific to the process.
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=9999    # TODO add in job ID
export WORLD_SIZE=$SLURM_NTASKS    # Note: only valid if ntasks==ngpus

# OMP THREADING
export OMP_NUM_THREADS=2  # OMP_NUM_THREADS=1 is the safe option ### Ville's script: OMP_NUM_THREADS is set to 1
export HSA_ENABLE_SDMA=0

# This setting is reported to provide a performance improvement
# (https://arxiv.org/pdf/2408.14090v1) but as of April 2025 is causing
# training instability on LUMI with pipeline parallelism.
# (see https://github.com/spyysalo/lumi-fineweb-replication/issues/1)
#export NCCL_NCHANNELS_PER_PEER=32

# Set interfaces to be used by RCCL.
# This is needed as otherwise RCCL tries to use a network interface it has
# no access to on LUMI.
export NCCL_SOCKET_IFNAME=hsn0,hsn1,hsn2,hsn3
export NCCL_NET_GDR_LEVEL=PHB
#export NCCL_DMABUF_ENABLE=1
export HSA_FORCE_FINE_GRAIN_PCIE=1

# DEBUGGING, INCREASE VERBOSITY IN LOGS
# Second script: commented out MIOPEN_ENABLE_LOGGING
export PYTHONWARNINGS=ignore
# export TORCH_SHOW_CPP_STACKTRACES=1
# export NCCL_DEBUG=INFO
# export RCCL_KERNEL_COLL_TRACE_ENABLE=1
# export NCCL_DEBUG_SUBSYS=ALL
# export NCCL_DEBUG_FILE=$OUTPUT_DIR/nccl-debug-${SLURM_JOB_NAME}-${SLURM_JOBID}.log #Move verbose nccl logging to its own file
export NVTE_DEBUG=0
export NVTE_DEBUG_LEVEL=0

# Second script: added a section on TORCH COMPILE/TORCHINDUCTOR variables
# Second script: added a section on LIBFABRIC variables
# Second script: added a section on NCCL/RCCL variables
# Second script: added NCCL_PROTO=Simple,LL
# Second script: added NCCL_CHECKS_DISABLE=1
# Second script: enabled NCCL_DMABUF_ENABLE=1
# Second script: added NCCL_TUNER_PLUGIN

######################################################################
#
# MODEL AND PRETRAINING CONFIGURATION
#
# This section sets variables that define the model and pretraining
# configuration. These mostly correspond to command-line arguments to
# Megatron-LM/pretrain_gpt.py, and when they do, the names should
# match (e.g. the variable $GLOBAL_BATCH_SIZE gets passed as
# --global-batch-size). This script is intended to be configurable by
# redefining these variables.
#
######################################################################

# huggingface exports
export HUGGING_FACE_HUB_TOKEN=$(cat ~/.huggingface/token)
export HF_HOME=$PWD/cache

# DATA
DATA_PATH="0.0295 /scratch/project_462000615/vitiugin/data_tok/deu/hplt2c-gemma-3/hplt2c_deu 0.0249 /scratch/project_462000615/vitiugin/data_tok/fra/hplt2c-gemma-3/hplt2c_fra 0.0309 /scratch/project_462000615/vitiugin/data_tok/spa/hplt2c-gemma-3/hplt2c_spa 0.0063 /scratch/project_462000615/vitiugin/data_tok/ces/hplt2c-gemma-3/hplt2c_ces 0.0027 /scratch/project_462000615/vitiugin/data_tok/dan/hplt2c-gemma-3/hplt2c_dan 0.0076 /scratch/project_462000615/vitiugin/data_tok/ell/hplt2c-gemma-3/hplt2c_ell 0.0035 /scratch/project_462000615/vitiugin/data_tok/fin/hplt2c-gemma-3/hplt2c_fin 0.0052 /scratch/project_462000615/vitiugin/data_tok/hun/hplt2c-gemma-3/hplt2c_hun 0.0140 /scratch/project_462000615/vitiugin/data_tok/ita/hplt2c-gemma-3/hplt2c_ita 0.0080 /scratch/project_462000615/vitiugin/data_tok/nld/hplt2c-gemma-3/hplt2c_nld 0.0129 /scratch/project_462000615/vitiugin/data_tok/pol/hplt2c-gemma-3/hplt2c_pol 0.0153 /scratch/project_462000615/vitiugin/data_tok/por/hplt2c-gemma-3/hplt2c_por 0.0050 /scratch/project_462000615/vitiugin/data_tok/ron/hplt2c-gemma-3/hplt2c_ron 0.0050 /scratch/project_462000615/vitiugin/data_tok/swe/hplt2c-gemma-3/hplt2c_swe 0.0022 /scratch/project_462000615/vitiugin/data_tok/bul/hplt2c-gemma-3/hplt2c_bul 0.0008 /scratch/project_462000615/vitiugin/data_tok/est/hplt2c-gemma-3/hplt2c_est 0.0010 /scratch/project_462000615/vitiugin/data_tok/hrv/hplt2c-gemma-3/hplt2c_hrv 0.0012 /scratch/project_462000615/vitiugin/data_tok/lit/hplt2c-gemma-3/hplt2c_lit 0.0006 /scratch/project_462000615/vitiugin/data_tok/lvs/hplt2c-gemma-3/hplt2c_lvs 0.0016 /scratch/project_462000615/vitiugin/data_tok/slk/hplt2c-gemma-3/hplt2c_slk 0.0008 /scratch/project_462000615/vitiugin/data_tok/slv/hplt2c-gemma-3/hplt2c_slv 0.0005 /scratch/project_462000615/vitiugin/data_tok/gle/hplt2c-gemma-3/hplt2c_gle 0.0005 /scratch/project_462000615/vitiugin/data_tok/mlt/hplt2c-gemma-3/hplt2c_mlt 0.0005 /scratch/project_462000615/vitiugin/data_tok/als/hplt2c-gemma-3/hplt2c_als 0.0010 /scratch/project_462000615/vitiugin/data_tok/bos/hplt2c-gemma-3/hplt2c_bos 0.0012 /scratch/project_462000615/vitiugin/data_tok/cat/hplt2c-gemma-3/hplt2c_cat 0.0005 /scratch/project_462000615/vitiugin/data_tok/eus/hplt2c-gemma-3/hplt2c_eus 0.0005 /scratch/project_462000615/vitiugin/data_tok/glg/hplt2c-gemma-3/hplt2c_glg 0.0005 /scratch/project_462000615/vitiugin/data_tok/isl/hplt2c-gemma-3/hplt2c_isl 0.0005 /scratch/project_462000615/vitiugin/data_tok/kat/hplt2c-gemma-3/hplt2c_kat 0.0005 /scratch/project_462000615/vitiugin/data_tok/mkd/hplt2c-gemma-3/hplt2c_mkd 0.0027 /scratch/project_462000615/vitiugin/data_tok/nob/hplt2c-gemma-3/hplt2c_nob 0.0001 /scratch/project_462000615/vitiugin/data_tok/nno/hplt2c-gemma-3/hplt2c_nno 0.0005 /scratch/project_462000615/vitiugin/data_tok/srp/hplt2c-gemma-3/hplt2c_srp 0.0077 /scratch/project_462000615/vitiugin/data_tok/tur/hplt2c-gemma-3/hplt2c_tur 0.0040 /scratch/project_462000615/vitiugin/data_tok/ukr/hplt2c-gemma-3/hplt2c_ukr 0.3431 /scratch/project_462000353/data/nemotron-cc/tokenized-gemma-3/medium-high-actual 0.3767 /scratch/project_462000353/data/nemotron-cc/tokenized-gemma-3/high-actual 0.0720 /scratch/project_462000615/data/starcoder/tokenized-gemma-3/starcoderdata 0.0080 /scratch/project_462000963/preprocessed/gemma-3/HuggingFaceTB/finemath/finemath-4plus-train"
DATA_CACHE_PATH="$BASE_DIR/cache"
TOKENIZER_MODEL="google/gemma-3-27b-pt"

# MODEL
NUM_LAYERS=32 # original value — 24
HIDDEN_SIZE=4096 # original value — 2048
FFN_HIDDEN_SIZE=14336 # original value — $((4*HIDDEN_SIZE))
NUM_ATTENTION_HEADS=32
NUM_QUERY_GROUPS=8 # original value - 32    # No GQA when NUM_QUERY_GROUPS=NUM_ATTENTION_HEADS
# original script  — TIE_WORD_EMBEDDINGS=1
INIT_METHOD_STD=0.00747017 # original script — 0.02
SEQ_LENGTH=2048
NUM_KV_HEADS=8 ### added in Ville's script

# PARALLELISM Ville's script: parallelism is passed as command-line arguments instead of variables.
PIPELINE_MODEL_PARALLEL_SIZE=1
TENSOR_MODEL_PARALLEL_SIZE=1
CONTEXT_PARALLEL_SIZE=1
NUM_LAYERS_PER_VIRTUAL_PIPELINE_STAGE=1
PROFILE=0

# OPTIMIZER
ADAM_BETA1=0.9
ADAM_BETA2=0.95
ADAM_EPS=1e-5 # original script:1e-8
LR=3e-5 # based on Slack converstaion. original script: 3e-4
MIN_LR=0
WARMUP_FRACTION=1/10
COOLDOWN_FRACTION=1/5
CLIP_GRAD=1.0
WEIGHT_DECAY=0.05

# TRAINING
FSDP=0
GLOBAL_BATCH_SIZE=2048
MICRO_BATCH_SIZE=2 # changed for 4TT training
RECOMPUTATION=0
TRAIN_TOKENS=4_000_000_000_000    # TRAIN_ITERS computed from this

# SAVING AND EVALUATION
LOG_INTERVAL=1
SAVE_INTERVAL=1000 # changed for 4TT run to 5000
EVAL_INTERVAL=2500 # changed for 4TT run to 10000
EVAL_ITERS=100

######################################################################
#
# DERIVED CONFIGURATION SETTINGS
#
# The following settings are derived from the configuration above.
# Do set these directly, as they will be overwritten here.
#
######################################################################

# Check that variables are not set (sanity)
confirm_unset() {
    local varname="$1"
    if [ -n "${!varname+x}" ]; then
    echo "Error: variable '$varname' should not be set." >&2
    exit 1
    fi
}
confirm_unset "TRAIN_ITERS"
confirm_unset "LR_DECAY_ITERS"
confirm_unset "LR_WSD_DECAY_ITERS"

divide_rounding_up() {
    echo $((($1+$2-1)/$2))
}

# Calculate TRAIN_ITERS from TRAIN_TOKENS
TRAIN_TOKENS=${TRAIN_TOKENS//_}    # drop "_" for bash math
ITER_TOKENS=$((SEQ_LENGTH * GLOBAL_BATCH_SIZE))
TRAIN_ITERS=$(divide_rounding_up $TRAIN_TOKENS $ITER_TOKENS)

# Set LR_WARMUP_ITERS and LR_WSD_DECAY_ITERS based on WARMUP_FRACTION
# and COOLDOWN_FRACTION
WARMUP_CALC=$((TRAIN_ITERS * ${WARMUP_FRACTION}))

# Cap warmup iterations at 25000 to prevent excessively long warmup periods
if [ $WARMUP_CALC -gt 25000 ]; then
   LR_WARMUP_ITERS=25000
else
   LR_WARMUP_ITERS=$WARMUP_CALC
fi

LR_WSD_DECAY_ITERS=$((TRAIN_ITERS*${COOLDOWN_FRACTION}))

# LR_DECAY_ITERS is simply set to TRAIN_ITERS
LR_DECAY_ITERS=$TRAIN_ITERS

######################################################################
#
# BUILDING COMMAND-LINE ARGUMENTS
#
# The following builds the command-line arguments for
# Megatron-LM/pretrain_gpt.py based on the variables defined above
# (and optionally in any config given to the script). Note that some
# arguments that are not expected to vary are hard-coded here.
#
######################################################################

DATA_ARGS=(
    --data-path "$DATA_PATH"
    --data-cache-path "$DATA_CACHE_PATH"
    --tokenizer-type HuggingFaceTokenizer
    --tokenizer-model "$TOKENIZER_MODEL"
    --make-vocab-size-divisible-by 128
    --dataloader-type single
    --num-workers 5   # Some issues with this, lower values are safer, in Ville's script it was changed to 5
)

MODEL_ARGS=(
    --num-layers $NUM_LAYERS
    --hidden-size $HIDDEN_SIZE
    --ffn-hidden-size $FFN_HIDDEN_SIZE
    --num-attention-heads $NUM_ATTENTION_HEADS
)

if [ "$NUM_QUERY_GROUPS" != "$NUM_ATTENTION_HEADS" ]; then
    MODEL_ARGS+=(
        --group-query-attention
        --num-query-groups $NUM_QUERY_GROUPS
    )
fi

if [ "$TIE_WORD_EMBEDDINGS" = "0" ]; then   ### This part is commented in Ville's version
    MODEL_ARGS+=(   ### This part is commented in Ville's version
    --untie-embeddings-and-output-weights   ### This part is commented in Ville's version
    )   ### This part is commented in Ville's version
fi  ### This part is commented in Ville's version

if [ "$FSDP" = "1" ]; then
    PARALLEL_ARGS=(
    --use-torch-fsdp2
    )
else
    PARALLEL_ARGS=(
    --tensor-model-parallel-size $TENSOR_MODEL_PARALLEL_SIZE
    --pipeline-model-parallel-size $PIPELINE_MODEL_PARALLEL_SIZE
    --context-parallel-size $CONTEXT_PARALLEL_SIZE  ### Removed in Ville's version
    --sequence-parallel
    --use-distributed-optimizer
    )
fi

if [ "$PROFILE" = "1" ]; then
    PROFILE_ARGS=(
    --use-pytorch-profiler
    --profile-ranks 0 # In Ville's script added $LAST_RANK: LAST_RANK=$((SLURM_NTASKS - 1))
    --profile-step-start 6 # original script: 5
    --profile-step-end 7
    )
else
    PROFILE_ARGS=()
fi

MODEL_ARGS+=(
    --use-flash-attn
    --attention-softmax-in-fp32
    --max-position-embeddings $SEQ_LENGTH
    --seq-length $SEQ_LENGTH
    --position-embedding-type rope
    --use-rotary-position-embeddings ### Original script: --rotary-base $ROTARY_BASE
    --disable-bias-linear
    --init-method-std $INIT_METHOD_STD
    --attention-dropout 0.0
    --hidden-dropout 0.0
    --normalization RMSNorm
    --micro-batch-size $MICRO_BATCH_SIZE
    --global-batch-size $GLOBAL_BATCH_SIZE
    --train-iters $TRAIN_ITERS
    --bf16
    --swiglu
    # Original sctipt: --no-async-tensor-model-parallel-allreduce
    --no-masked-softmax-fusion
    --no-gradient-accumulation-fusion
    # Original sctipt: --no-bias-dropout-fusion
    --no-rope-fusion    # buggy on AMD, do not enable without validating
    --distributed-timeout-minutes 30
    --overlap-grad-reduce
    --overlap-param-gather
    --accumulate-allreduce-grads-in-fp32
)

OPTIMIZER_ARGS=(
    --optimizer adam
    --adam-beta1 $ADAM_BETA1
    --adam-beta2 $ADAM_BETA2
    --adam-eps $ADAM_EPS
    --lr $LR
    --min-lr $MIN_LR
    --lr-decay-style "WSD"
    --lr-wsd-decay-style "linear"
    --lr-warmup-iters $LR_WARMUP_ITERS
    --lr-decay-iters $LR_DECAY_ITERS
    --lr-wsd-decay-iters $LR_WSD_DECAY_ITERS
    --clip-grad $CLIP_GRAD
    --weight-decay $WEIGHT_DECAY
    --use-distributed-optimizer ## Added in Ville's sxcript
)

OUTPUT_ARGS=(
    --eval-interval $EVAL_INTERVAL
    --eval-iters $EVAL_ITERS
    --tensorboard-dir "$TENSORBOARD_DIR"
    --tensorboard-queue-size 5
    --wandb-project "$WANDB_PROJECT"
    --wandb-exp-name "$WANDB_EXP_NAME"
    --wandb-save-dir "$WANDB_DIR"
    --log-throughput
    --log-progress
    --log-interval $LOG_INTERVAL
)
# Second script: removed --log-progress
# Second script: added --log-timers-to-tensorboard, --log-params-norm, and --no-one-logger
# Second script: output args are conditionally added if ENABLE_WANDB is 1

# Interleaved pipeline scheduling is only possible with pipeline
# parallel degree > 1.
if [ $PIPELINE_MODEL_PARALLEL_SIZE -gt 1 ] && [ $NUM_LAYERS_PER_VIRTUAL_PIPELINE_STAGE -gt 1 ]; then
    PARALLEL_ARGS+=(
    --num-layers-per-virtual-pipeline-stage $NUM_LAYERS_PER_VIRTUAL_PIPELINE_STAGE
    )
fi

if [ "$RECOMPUTATION" = "1" ]; then
    MODEL_ARGS+=(
    --recompute-activations
    --recompute-granularity selective
    )
fi

CHECKPOINT_ARGS=(
    --ckpt-format torch    # "legacy" checkpoints; torch_dist is crashing
#     --async-save    # requires --ckpt-format torch_dist
    --load "$CHECKPOINT_PATH" # Ville's script: removes --load
    --save "$CHECKPOINT_PATH"
    --save-interval $SAVE_INTERVAL
)

COMMAND=" \
    Megatron-LM/pretrain_gpt.py \
    "${MODEL_ARGS[@]}" \
    "${OPTIMIZER_ARGS[@]}" \
    "${PARALLEL_ARGS[@]}" \
    "${OUTPUT_ARGS[@]}" \
    "${CHECKPOINT_ARGS[@]}" \
    "${DATA_ARGS[@]}" \
    "${PROFILE_ARGS[@]}" \
"
# Second script: builds the command string differently by concatenating argument strings instead of arrays

######################################################################
#
# Run the command through the launch script with srun.
# Note that any node-specific setup needs to go into the launch script.
#
######################################################################

echo '============= COMMAND: ============='
echo "$COMMAND"
echo '===================================='

echo "START $SLURM_JOBID: $(date)"
echo "SLURM_NNODES: $SLURM_NNODES"
echo "SLURM_CPUS_PER_TASK: $SLURM_CPUS_PER_TASK"

srun \
    --label \
    --cpu-bind=mask_cpu:$BIND_MASK \
    singularity exec \
    -B "$BASE_DIR" \
    -B "$BIND_DIRS" \
    "$CONTAINER" \
    "$LAUNCH_SCRIPT" \
    $COMMAND

echo "END $SLURM_JOBID: $(date)"
# Second script: added 'wd' variable and used it to bind the current directory
# Second script: also added 'echo' statements to print container and working directory