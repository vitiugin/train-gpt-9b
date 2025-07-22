#!/bin/bash
#SBATCH --job-name=8B-64N-LR
#SBATCH --cpus-per-task=7
#SBATCH --ntasks-per-node=8
#SBATCH --nodes=64
#SBATCH --mem=480G
#SBATCH --partition=standard-g
#SBATCH --time=47:59:00
#SBATCH --exclusive
#SBATCH --gpus-per-node=8
#SBATCH --account=project_462000353                    #Insert your own project number here
#SBATCH --output logs/%j.out
#SBATCH --error logs/%j.err

set -eox pipefail
echo "Starting bash script"
module purge
#Save the current working directory for shorter path variables
wd=(`pwd`)

#COMPILER
#This is for the c++ dataset helper
export CC=gcc-12
export CXX=g++-12

# parsing input arguments
for ARGUMENT in "$@"
do
   KEY=$(echo $ARGUMENT | cut -f1 -d=)

   KEY_LENGTH=${#KEY}
   VALUE="${ARGUMENT:$KEY_LENGTH+1}"

   export "$KEY"="$VALUE"
done

#DISTRIBUTED ARGS
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=9999
export CUDA_DEVICE_MAX_CONNECTIONS=1                    #This is needed for sequence paralellism

#OMP THREADING
export OMP_NUM_THREADS=1                                #Virtual threads to accompany the HW threads set by slurm

#HSA=Heterogeneous System Architecture
#AMD provides HSA through ROCR https://rocm.docs.amd.com/projects/ROCR-Runtime/en/docs-6.2.4/index.html#hsa-runtime-api-and-runtime-for-rocm
#export HSA_ENABLE_SDMA=1                                #https://gpuopen.com/learn/amd-lab-notes/amd-lab-notes-gpu-aware-mpi-readme/
                                                        #https://rocm.docs.amd.com/en/docs-6.2.0/conceptual/gpu-memory.html#system-direct-memory-access
                                                        #HSA_ENABLE_SDMA=0 -- Uses blit kernels for comms through the gpu -- more bandwith between gcd's
                                                        #With HSA_ENABLE_SDMA=1 capped at 50GB/S uni-directional bandwidth
                                                        #Enables overlapping communication with computation
#export HSA_ENABLE_PEER_SDMA=0                          #Enable or disable SDMA engines for device-to-device copies
                                                        #https://rocm.docs.amd.com/projects/ROCR-Runtime/en/docs-6.2.4/api-reference/environment_variables.html
#export HSA_XNACK=1                                     #A fallback for pagefault https://rocm.docs.amd.com/en/docs-6.2.4/conceptual/gpu-memory.html#xnack

export PYTHONWARNINGS=ignore
#####
#DEBUGGING, INCREASE VERBOSITY IN LOGS

##TORCH##
#export TORCH_SHOW_CPP_STACKTRACES=1                    #C++ traceback for pytorch crashes
#export TORCH_DIST_INIT_BARRIER=1                       #Initialize torch distributed new group instance with a barrier to synchronize ranks

##TORCH COMPILE/TORCHINDUCTOR##
export TORCH_COMPILE_DISABLE=1
#export TORCH_CUDA_ARCH_LIST=gfx90a
export TORCHINDUCTOR_FORCE_DISABLE_CACHES=1
export TORCHINDUCTOR_UNIQUE_KERNEL_NAMES=1
export TORCHINDUCTOR_COMPILE_THREADS=5                  #I'm thinking we dont wanna use all of the cpu threads for compiling kernels
export TORCH_LOGS="-inductor,-aot"

##MIOPEN##
#MIOPEN is what rocm uses for some deep learning primitives; some of the normalization and stuff, mostly associated with CNN's and RNN's
#export MIOPEN_ENABLE_LOGGING=1
#export MIOPEN_LOG_LEVEL=5

##TORCHNCCL##
#TORCH FLIGHTRECORDER, see https://pytorch.org/tutorials/prototype/flight_recorder_tutorial.html
#export TORCH_NCCL_TRACE_BUFFER_SIZE=2000
#export TORCH_NCCL_DUMP_ON_TIMEOUT=true
#export TORCH_NCCL_DEBUG_INFO_TEMP_FILE=logs/fr/nccl_trace_rank

##LIBFABRIC##
#export FI_LOG_LEVEL=debug                              #Very verbose Libfabric logging. May cause performance degredation when set to 'debug'
#export FI_LOG_PROV=cxi                                 #Determine which libfabric provider to get logs from
#export FI_CXI_ODP=1                                    #On demand paging. This should be an alternative to the CXI_FORK_SAFE for unsafe fork. See more with man intro_mpi
#export FI_HMEM_DISABLE_P2P=1
export FI_HMEM=rocr                                     #Ensure that libfabric uses rocr's hmem instead of system, which is xpmem
export FI_HMEM_ROCR_USE_DMABUF=1
export FI_MR_ROCR_CACHE_MONITOR_ENABLED=1               #Detecting ROCR device memory (FI_HMEM_ROCR) changes made between the device virtual addresses used by an application and the underlying device physical pages.
#export FI_MR_CACHE_MONITOR=disabled
#export SLINGSHOT_DEVICES=cxi0,cxi1                     #Specify which cxi NIC's get enabled
#export FI_CXI_RDZV_GET_MIN=16364                       #Min byte size of message for cxi to use rendezvouz instead of eager messaging
#export FI_CXI_RDZV_THRESHOLD=32728                     #Max byte size
#export FI_CXI_DEFAULT_TX_SIZE=4096                     #Default is 1024
#export FI_CXI_RX_SIZE=4096
#export FI_CXI_RX_MATCH_MODE=hybrid                      #Address matching can be either hardware(cxi),software or hybrid
export FI_MR_CACHE_MONITOR=userfaultfd                  #https://support.hpe.com/hpesc/public/docDisplay?docId=dp00005991en_us&page=user/memory_cache_monitor_settings.html#ariaid-title1
export FI_CXI_DEFAULT_CQ_SIZE=131072                    #From HPE whitepaper, default value is only 1024
#export FI_CXI_DEFAULT_TX_SIZE=2048
#export FI_CXI_RDZV_PROTO=alt_read                       #Alternative rendezvous protocol developed by HPE. https://docs.olcf.ornl.gov/software/analytics/pytorch_frontier.html#alternative-rendezvous-protocol
#export FI_CXI_REQ_BUF_SIZE=
#export FI_CXI_DISABLE_HMEM_DEV_REGISTER

##NCCL/RCCL##
#export RCCL_KERNEL_COLL_TRACE_ENABLE=1                 #Print c++ code traceback for nccl collectives
#export NCCL_DEBUG=INFO                                 #Verbose logging from RCCL
#export NCCL_DEBUG_SUBSYS=INIT,ENV,TRACE,COLL           #Specify which stuff is logged from NCCL_DEBUG=INFO. Choices are P2P(Peer2Peer), COLL(Collecetives) etc.
#export NCCL_DEBUG_FILE=logs/nccl-$SLURM_JOB_NAME.log   #Move verbose nccl logging to its own file
#export NCCL_NCHANNELS_PER_PEER=2                       #Increases the amount of channels available for p2p communication. Should be a slight performance boost
#export NCCL_MAX_NCHANNELS=16
#export NCCL_MAX_NCHANNELS=16
#export HIP_LAUNCH_BLOCKING=1                           #MAKES NCCL COLLECTIVES SYNCHRONOUS INSTEAD OF ASYNC
export NCCL_PROTO=Simple,LL                             #Simple or LL instead of ll128
export NCCL_CHECKS_DISABLE=1                            #Disable checks before collectives, which increases performance
export NCCL_DMABUF_ENABLE=1                            #Enable DMA buffers from the nccl side
#export GPU_MAX_HW_QUEUES=2
export NCCL_SET_THREAD_NAME=1                           #Give meaningful names to NCCL cpu threads


##HIP/AMD##
#export AMD_SERIALIZE_KERNEL=3                          #Verbose logs from HIP
#export AMD_LOG_LEVEL=1                                 #Very verbose logging. In our context seems to concern hipified cuda kernels

##TRITON##
export TRITON_ALWAYS_COMPILE=1                         #Always force Triton to compile even in the case of cache hit.
#export TRITON_CACHE_DIR=/scratch/project_462000615/triton_cache

##TransformerEngine##
#export NVTE_DEBUG=1                                    #Shows things like what dot product attention implementation is being used
#export NVTE_DEBUG_LEVEL=2
#export NVTE_USE_RMSNORM_TRITON=1                        #Experimental RMRSNORM triton kernel optimized for AMD/ROCM
#export GEMM_TUNING=1
#export TE_HIPBLASLT_TUNING_RUN_COUNT=10
#export TE_HIPBLASLT_TUNING_ALGO_COUNT=50

BASE_DIR="$SLURM_SUBMIT_DIR"

# sets TRAIN_DATA and VALIDATION_DATA
#TRAIN_DATA=/scratch/project_462000963/preprocessed/gemma-3/nemotron-cc/medium-actual-100BT
TRAIN_DATA="0.0445 /scratch/project_462000615/vitiugin/data_tok/deu/hplt2c-gemma-3/hplt2c_deu 0.0376 /scratch/project_462000615/vitiugin/data_tok/fra/hplt2c-gemma-3/hplt2c_fra 0.0467 /scratch/project_462000615/vitiugin/data_tok/spa/hplt2c-gemma-3/hplt2c_spa 0.0094 /scratch/project_462000615/vitiugin/data_tok/ces/hplt2c-gemma-3/hplt2c_ces 0.0041 /scratch/project_462000615/vitiugin/data_tok/dan/hplt2c-gemma-3/hplt2c_dan 0.0114 /scratch/project_462000615/vitiugin/data_tok/ell/hplt2c-gemma-3/hplt2c_ell 0.0053 /scratch/project_462000615/vitiugin/data_tok/fin/hplt2c-gemma-3/hplt2c_fin 0.0078 /scratch/project_462000615/vitiugin/data_tok/hun/hplt2c-gemma-3/hplt2c_hun 0.0212 /scratch/project_462000615/vitiugin/data_tok/ita/hplt2c-gemma-3/hplt2c_ita 0.0122 /scratch/project_462000615/vitiugin/data_tok/nld/hplt2c-gemma-3/hplt2c_nld 0.0194 /scratch/project_462000615/vitiugin/data_tok/pol/hplt2c-gemma-3/hplt2c_pol 0.0231 /scratch/project_462000615/vitiugin/data_tok/por/hplt2c-gemma-3/hplt2c_por 0.0076 /scratch/project_462000615/vitiugin/data_tok/ron/hplt2c-gemma-3/hplt2c_ron 0.0075 /scratch/project_462000615/vitiugin/data_tok/swe/hplt2c-gemma-3/hplt2c_swe 0.0033 /scratch/project_462000615/vitiugin/data_tok/bul/hplt2c-gemma-3/hplt2c_bul 0.0012 /scratch/project_462000615/vitiugin/data_tok/est/hplt2c-gemma-3/hplt2c_est 0.0015 /scratch/project_462000615/vitiugin/data_tok/hrv/hplt2c-gemma-3/hplt2c_hrv 0.0018 /scratch/project_462000615/vitiugin/data_tok/lit/hplt2c-gemma-3/hplt2c_lit 0.0010 /scratch/project_462000615/vitiugin/data_tok/lvs/hplt2c-gemma-3/hplt2c_lvs 0.0024 /scratch/project_462000615/vitiugin/data_tok/slk/hplt2c-gemma-3/hplt2c_slk 0.0012 /scratch/project_462000615/vitiugin/data_tok/slv/hplt2c-gemma-3/hplt2c_slv 0.0005 /scratch/project_462000615/vitiugin/data_tok/gle/hplt2c-gemma-3/hplt2c_gle 0.0005 /scratch/project_462000615/vitiugin/data_tok/mlt/hplt2c-gemma-3/hplt2c_mlt 0.0006 /scratch/project_462000615/vitiugin/data_tok/als/hplt2c-gemma-3/hplt2c_als 0.0015 /scratch/project_462000615/vitiugin/data_tok/bos/hplt2c-gemma-3/hplt2c_bos 0.0018 /scratch/project_462000615/vitiugin/data_tok/cat/hplt2c-gemma-3/hplt2c_cat 0.0005 /scratch/project_462000615/vitiugin/data_tok/eus/hplt2c-gemma-3/hplt2c_eus 0.0005 /scratch/project_462000615/vitiugin/data_tok/glg/hplt2c-gemma-3/hplt2c_glg 0.0005 /scratch/project_462000615/vitiugin/data_tok/isl/hplt2c-gemma-3/hplt2c_isl 0.0005 /scratch/project_462000615/vitiugin/data_tok/kat/hplt2c-gemma-3/hplt2c_kat 0.0005 /scratch/project_462000615/vitiugin/data_tok/mkd/hplt2c-gemma-3/hplt2c_mkd 0.0040 /scratch/project_462000615/vitiugin/data_tok/nob/hplt2c-gemma-3/hplt2c_nob 0.0002 /scratch/project_462000615/vitiugin/data_tok/nno/hplt2c-gemma-3/hplt2c_nno 0.0006 /scratch/project_462000615/vitiugin/data_tok/srp/hplt2c-gemma-3/hplt2c_srp 0.0116 /scratch/project_462000615/vitiugin/data_tok/tur/hplt2c-gemma-3/hplt2c_tur 0.0060 /scratch/project_462000615/vitiugin/data_tok/ukr/hplt2c-gemma-3/hplt2c_ukr 0.3003 /scratch/project_462000353/data/nemotron-cc/tokenized-gemma-3/medium-high-actual 0.3297 /scratch/project_462000353/data/nemotron-cc/tokenized-gemma-3/high-actual 0.0630 /scratch/project_462000615/data/starcoder/tokenized-gemma-3/starcoderdata 0.0070 /scratch/project_462000963/preprocessed/gemma-3/HuggingFaceTB/finemath/finemath-4plus-train"

DATA_CACHE=/scratch/project_462000615/vitiugin/8b_cache/
#MERGES=/scratch/project_462000615/villekom/reproducible/merges.txt
#VOCAB=/scratch/project_462000615/villekom/reproducible/vocab.json

# huggingface exports
export HUGGING_FACE_HUB_TOKEN=$(cat ~/.huggingface/token)
export HF_HOME=$PWD/cache

TOKENIZER_MODEL="google/gemma-3-27b-pt"

#Optional cli arguments
LR="${LR:-1.5e-4}"
MIN_LR="${MIN_LR:-0}"
MODEL_SIZE="${MODEL_SIZE:-8}"
TP_SIZE="${TP:-2}"
PP_SIZE="${PP:-1}"
VPP_SIZE="${VPP_SIZE:-1}"
NN="${NN:-$SLURM_NNODES}"
N_tasks="${N_tasks:-$SLURM_NTASKS}"
#TRAIN_ITERS="${TRAIN_ITERS:-10}"
SEQ_LEN="${SEQ_LEN:-4096}"
ENABLE_PROFILING="${ENABLE_PROFILING:-0}" #enable pytorch profiling
#WANDB_PROJECT="${WANDB_PROJECT:-"debug"}"

# WEIGHTS & BIASES CONFIG
WANDB_PROJECT="oelln-8B-4T"
WANDB_EXP_NAME="8B-test"

WARMUP_FRACTION=1/10
COOLDOWN_FRACTION=1/5


##BATCH SIZE
#GLOBAL_BATCH_SIZE="${GBS:-1024}"
#MBS max 2 with the default model and model parallel parameters
#MICRO_BATCH_SIZE="${MBS:-2}"

GLOBAL_BATCH_SIZE=2048  # changed for 64N 4TT training
MICRO_BATCH_SIZE=2 # changed for 64N 4TT training

TRAIN_TOKENS=4_000_000_000_000    # TRAIN_ITERS computed from this
divide_rounding_up() {
    echo $((($1+$2-1)/$2))
}

# Calculate TRAIN_ITERS from TRAIN_TOKENS
TRAIN_TOKENS=${TRAIN_TOKENS//_}    # drop "_" for bash math
ITER_TOKENS=$((SEQ_LEN * GLOBAL_BATCH_SIZE))
TRAIN_ITERS=$(divide_rounding_up $TRAIN_TOKENS $ITER_TOKENS)

# Set LR_WARMUP_ITERS and LR_WSD_DECAY_ITERS based on WARMUP_FRACTION
# and COOLDOWN_FRACTION
#LR_WARMUP_ITERS=$((TRAIN_ITERS*${WARMUP_FRACTION}))
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

#LLama 34B
if [[ $MODEL_SIZE -eq 34 ]]; then
    MODEL=EUROPA-34B
    NLAYERS=56
    NHIDDEN=7168
    NHEADS=56
    FFN_HIDDEN_SIZE=20480
    SEQ_LEN=4096
    NUM_KV_HEADS=8
    NUM_QUERY_GROUPS=8
elif [[ $MODEL_SIZE -eq 8 ]]; then
    MODEL=LLAMA3_8B
    NHIDDEN=4096
    FFN_HIDDEN_SIZE=14336
    NLAYERS=32
    NHEADS=32
    SEQ_LEN=2048
    NUM_KV_HEADS=32
    GROUP_SIZE=1
    NUM_QUERY_GROUPS=32
fi

#TENSORBOARD_PATH=/scratch/project_462000615/$USER/logs/megLM/$SLURM_JOB_NAME
#WANDB_SAVE_DIR=/scratch/project_462000615/$USER/logs/megLM/wandb/$SLURM_JOB_NAME
#CHECKPOINT_PATH=/scratch/project_462000615/$USER/ckpts/megLM/$SLURM_JOB_NAME

OUTPUT_DIR="/scratch/project_462000963/users/vitiugin/checkpoints_megatron/4T-8B"
CHECKPOINT_PATH="$OUTPUT_DIR/checkpoints"
TENSORBOARD_DIR="$OUTPUT_DIR/tensorboard/$SLURM_JOB_NAME-$SLURM_JOBID"
WANDB_SAVE_DIR="$OUTPUT_DIR/wandb"


LOG_INTERVAL=1
SAVE_INTERVAL=1000
EVAL_INTERVAL=2000
EVAL_ITERS=100
INIT_METHOD_STD=0.00747017

OPTIMIZER_ARGS=" \
    --optimizer adam \
    --adam-beta1 0.9 \
    --adam-beta2 0.95 \
    --adam-eps 1e-5 \
    --use-distributed-optimizer \
    --lr $LR \
    --min-lr $MIN_LR \
    --lr-decay-style WSD \
    --lr-wsd-decay-style linear \
    --lr-warmup-iters $LR_WARMUP_ITERS \
    --lr-decay-iters $LR_DECAY_ITERS \
    --lr-wsd-decay-iters $LR_WSD_DECAY_ITERS \
    --clip-grad 1.0 \
    --weight-decay 0.05 \
    "

LAST_RANK=$((SLURM_NTASKS - 1))

#PYTORCH PROFILER ARGS
PROFILE_ARGS=" \
    --profile \
    --use-pytorch-profiler \
    --profile-step-start 6 \
    --profile-step-end 7 \
    --profile-ranks 0 $LAST_RANK \
    --tensorboard-dir $TENSORBOARD_PATH \
    --tensorboard-queue-size 5 \
    "

DATA_ARGS=" \
    --data-path $TRAIN_DATA \
    --data-cache-path $DATA_CACHE \
    --tokenizer-type HuggingFaceTokenizer \
    --tokenizer-model $TOKENIZER_MODEL \
    --make-vocab-size-divisible-by 128 \
    --dataloader-type single \
    --num-workers 5 \
    "

GPT_ARGS=" \
    --num-layers $NLAYERS \
    --hidden-size $NHIDDEN \
    --num-attention-heads $NHEADS \
    --ffn-hidden-size $FFN_HIDDEN_SIZE \
    --max-position-embeddings $SEQ_LEN \
    --seq-length $SEQ_LEN \
    --train-iters $TRAIN_ITERS \
    --eval-iters $EVAL_ITERS \
    --micro-batch-size $MICRO_BATCH_SIZE \
    --global-batch-size $GLOBAL_BATCH_SIZE \
    --tokenizer-type GPT2BPETokenizer \
    --normalization RMSNorm \
    --bf16 \
    --init-method-std $INIT_METHOD_STD \
    --seed 42 \
    --swiglu \
    --attention-dropout 0 \
    --hidden-dropout 0 \
    --attention-softmax-in-fp32 \
    --accumulate-allreduce-grads-in-fp32 \
    --use-rotary-position-embeddings \
    --group-query-attention \
    --num-query-groups $NUM_QUERY_GROUPS \
    --distributed-timeout-minutes 10 \
    --use-flash-attn \
    --overlap-grad-reduce \
    --overlap-param-gather \
    --save $CHECKPOINT_PATH \
    --save-interval $SAVE_INTERVAL \
    --ckpt-format torch \
    "

FUSED_KERNEL_ARGS=" \
    --no-gradient-accumulation-fusion \
    --disable-bias-linear \
    --no-masked-softmax-fusion \
    --no-rope-fusion \
    "

OUTPUT_ARGS=" \
    --log-throughput \
    --log-timers-to-tensorboard \
    --log-params-norm \
    --log-interval 1 \
    --logging-level 40 \
    --wandb-save-dir $WANDB_SAVE_DIR \
    --wandb-project $WANDB_PROJECT \
    --wandb-exp-name $SLURM_JOB_NAME \
    "

PARALLEL_ARGS="\
    --tensor-model-parallel-size $TP_SIZE \
    --pipeline-model-parallel-size $PP_SIZE \
    --sequence-parallel \
"

if (( VPP_SIZE > 1)); then
    PARALLEL_ARGS="$PARALLEL_ARGS \
    --num-layers-per-virtual-pipeline-stage $VPP_SIZE"
fi

CMD=" \
    $BASE_DIR/Megatron-LM/pretrain_gpt.py \
    $GPT_ARGS \
    $PARALLEL_ARGS \
    $OUTPUT_ARGS \
    $OPTIMIZER_ARGS \
    $DATA_ARGS \
    $FUSED_KERNEL_ARGS \
    "

if [ "$ENABLE_PROFILING" -eq 1 ]; then
    CMD+=" $PROFILE_ARGS"
fi

c="fe"

# Bind mask for one hardware thread per core
# Set hardware threads on slurm with --threads-per-core. Default is 1.
BIND_MASK="0x${c}000000000000,0x${c}00000000000000,0x${c}0000,0x${c}000000,0x${c},0x${c}00,0x${c}00000000,0x${c}0000000000"

echo "START $SLURM_JOBID: $(date)"
echo "NNODES" $SLURM_NNODES
echo "CPUS PER TASK" $SLURM_CPUS_PER_TASK

#SINGULARITY ENVIRONMENT
export CONTAINER=/appl/local/containers/sif-images/lumi-pytorch-rocm-6.2.4-python-3.12-pytorch-v2.6.0.sif
export SINGULARITY_BIND=/boot,/pfs,/scratch,/projappl,/project,/flash,/appl,/opt/cray,/var/spool/slurmd
echo "CONTAINER" $CONTAINER

export PWD=(`pwd -P`)

srun --label --cpu-bind=mask_cpu:$BIND_MASK \
    singularity exec \
    -B $PWD \
    $CONTAINER \
    $BASE_DIR/debug_launch.sh \
    $CMD

echo "END $SLURM_JOBID: $(date)"