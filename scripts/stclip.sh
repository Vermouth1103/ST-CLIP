#!/bin/bash

# custom config
DATA=path/to/data
TRAINER=STCLIP

DATASET=$1
CFG=$2  # config file
SHOTS=$3  # number of shots (1, 2, 4, 8, 16)

for SEED in 1 2 3
do
    DIR=output/${DATASET}/stclip/${CFG}_${SHOTS}shots/nctx${NCTX}_csc${CSC}_ctp${CTP}/seed${SEED}
    if [ -d "$DIR" ]; then
        echo "Oops! The results exist at ${DIR} (so skip this job)"
    else
        python train.py \
        --root ${DATA} \
        --seed ${SEED} \
        --trainer ${TRAINER} \
        --dataset-config-file configs/datasets/${DATASET}.yaml \
        --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
        --output-dir ${DIR} \
        DATASET.NUM_SHOTS ${SHOTS}
    fi
done

# e.g.
# bash scripts/stclip.sh beijing_dataset vit_b32_ep50 16