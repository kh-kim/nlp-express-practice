BACKBONE=$1 # e.g. "klue/roberta-base"

# replace / with _ in BACKBONE
MODEL_NAME="nsmc-${BACKBONE//\//-}"
TRAIN_TSV_FN="./data/ratings_train.tsv"
VALID_TSV_FN="./data/ratings_valid.tsv"
TEST_TSV_FN="./data/ratings_test.tsv"

python finetune.py \
    --model_name $MODEL_NAME \
    --train_tsv_fn $TRAIN_TSV_FN \
    --valid_tsv_fn $VALID_TSV_FN \
    --test_tsv_fn $TEST_TSV_FN \
    --backbone $BACKBONE \
    --num_train_epochs 3 \
    --batch_size_per_device 64 \
    --gradient_accumulation_steps 4 \
    --fp16 \
    --max_length 128 \
