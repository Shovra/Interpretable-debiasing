# lamda [1, 0.1, 0.01, 0.001, 0.0001]
# selection [0.3, 0.5, 0.7]
# lasso [1, 0.1, 0.01, 0.001, 0.0001]
BS=512
SELECT_NAME=03
OUT_DIR=jigsaw_results/gender/origin/latent
LOG_DIR=jigsaw_results/gender/origin/latent
declare -A CUDAS=( [0]=1 [1]=01 [2]=001 [3]=0001)
declare -A SELECT=( [03]=0.3 [05]=0.5 [07]=0.7)
declare -A LASSO=( [1]=1 [01]=0.1 [001]=0.01 [0001]=0.001)
declare -A LAMBDA=( [1]=1 [01]=0.1 [001]=0.01 [0001]=0.001 [00001]=0.0001)

CUDA_ID=0
LS=${LASSO[${CUDAS[$CUDA_ID]}]}
for lambda in 1 01 001 0001 00001; do
    LV=${LAMBDA[$lambda]}
    SV=${SELECT[$SELECT_NAME]}

    NAME=pct$SELECT_NAME.lmd$lambda.lasso${CUDAS[$CUDA_ID]}.z
    mkdir -p $OUT_DIR/$NAME
    echo "==============Now run $NAME on CUDA $CUDA_ID=============="
    echo "$OUT_DIR/$NAME/"
    CUDA_VISIBLE_DEVICES=$CUDA_ID python train_single.py --model latent \
        --save_path $OUT_DIR/$NAME/ \
        --batch_size $BS \
        --dependent-z \
        --selection  $SV \
        --lasso $LS \
        --lambda_init $LV > $LOG_DIR/$NAME/log.log &
done



CUDA_ID=1
LS=${LASSO[${CUDAS[$CUDA_ID]}]}
for lambda in 1 01 001 0001 00001; do
    LV=${LAMBDA[$lambda]}
    SV=${SELECT[$SELECT_NAME]}

    NAME=pct$SELECT_NAME.lmd$lambda.lasso${CUDAS[$CUDA_ID]}.z
    mkdir -p $OUT_DIR/$NAME

    echo "==============Now run $NAME on CUDA $CUDA_ID=============="
    echo "$OUT_DIR/$NAME/"
    CUDA_VISIBLE_DEVICES=$CUDA_ID python train_single.py --model latent \
        --save_path $OUT_DIR/$NAME/ \
        --batch_size $BS \
        --dependent-z \
        --selection  $SV \
        --lasso $LS \
        --lambda_init $LV > $LOG_DIR/$NAME/log.log &
done



CUDA_ID=2
LS=${LASSO[${CUDAS[$CUDA_ID]}]}
for lambda in 1 01 001 0001 00001; do
    LV=${LAMBDA[$lambda]}
    SV=${SELECT[$SELECT_NAME]}

    NAME=pct$SELECT_NAME.lmd$lambda.lasso${CUDAS[$CUDA_ID]}.z
    mkdir -p $OUT_DIR/$NAME
    echo "==============Now run $NAME on CUDA $CUDA_ID=============="
    echo "$OUT_DIR/$NAME/"
    CUDA_VISIBLE_DEVICES=$CUDA_ID python train_single.py --model latent \
        --save_path $OUT_DIR/$NAME/ \
        --batch_size $BS \
        --dependent-z \
        --selection  $SV \
        --lasso $LS \
        --lambda_init $LV > $LOG_DIR/$NAME/log.log &
done



CUDA_ID=3
LS=${LASSO[${CUDAS[$CUDA_ID]}]}
for lambda in 1 01 001 0001 00001; do
    LV=${LAMBDA[$lambda]}
    SV=${SELECT[$SELECT_NAME]}

    NAME=pct$SELECT_NAME.lmd$lambda.lasso${CUDAS[$CUDA_ID]}.z
    mkdir -p $OUT_DIR/$NAME
    echo "==============Now run $NAME on CUDA $CUDA_ID=============="
    echo "$OUT_DIR/$NAME/"
    CUDA_VISIBLE_DEVICES=$CUDA_ID python train_single.py --model latent \
        --save_path $OUT_DIR/$NAME/ \
        --batch_size $BS \
        --dependent-z \
        --selection  $SV \
        --lasso $LS \
        --lambda_init $LV > $LOG_DIR/$NAME/log.log &
done