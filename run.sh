#!/bin/bash


DATASET=$1
METHOD=$2
NOTE=$3
TRMODE=$4
EVALMODE=$5
USE_RAND_LOADER=$6
SAMP_STRATEGY=$7
USE_TF=$8
DA_STRATEGY=$9


ROOTDIR="./output/${DATASET}/neuralese/${TRMODE}/${METHOD}/${NOTE}/${DA_STRATEGY}"
SUBDIR="/${EVALMODE}_us-rd_ld-${USE_RAND_LOADER}_${SAMP_STRATEGY}_${DATASET}"

CPTDIR="${ROOTDIR}/checkpoints/${SUBDIR}/"
RESULTDIR="${ROOTDIR}/results/${SUBDIR}/"
TBDIR="${ROOTDIR}/tensorboard_logs/${SUBDIR}/"
LOG="${ROOTDIR}/logs/${SUBDIR}.txt.`date +'%Y-%m-%d_%H-%M-%S'`.log"
mkdir -p "${ROOTDIR}/logs"
mkdir -p ${CPTDIR}
mkdir -p ${RESULTDIR}
mkdir -p ${TBDIR}

exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

CUDA_VISIBLE_DEVICES=0 python tasks/R2R-pano/main.py \
    --exp_name ${NOTE} \
    --batch_size 64 \
    --img_fc_use_angle 1 \
    --img_feat_input_dim 2176 \
    --img_fc_dim 1024 \
    --rnn_hidden_size 512 \
    --eval_every_epochs 5 \
    --use_ignore_index 1 \
    --arch ${METHOD} \
    --fix_action_ended 0 \
    --use_end_token 0 \
    --results_dir ${RESULTDIR} \
    --checkpoint_dir ${CPTDIR} \
    --log_dir ${TBDIR} \
    --training_mode ${TRMODE} \
    --evaluation_mode ${EVALMODE} \
    --use_random_loader ${USE_RAND_LOADER} \
    --sampling_strategy ${SAMP_STRATEGY} \
    --dataset_name ${DATASET} \
    --use_teacher_forcing ${USE_TF} \
    --speaker_DA_strategy ${DA_STRATEGY} \

echo 'Done'

# bash run_neuralese_agent.sh R2R cogrounding test_proxy_neuralese
