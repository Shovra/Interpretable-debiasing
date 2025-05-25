# interpretable_debiasing
Source code for paper [Controlling Bias Exposure for Fair Interpretable Predictions](https://arxiv.org/abs/2210.07455) [EMNLP2022]


## Prepare:
1. Download `glove.840B.300d.sst.txt` to the current folder. 
2. Create a `datasets/` folder and prepare your data in `datasets/`. You can also download our data  from [GDrive](https://drive.google.com/drive/folders/1ClCTyogTPWD86tvNPJk8IGRR9Sc4NSU6?usp=sharing).


We support the following datasets:


1. Jigsaw toxicity classification: The outcome is the toxicity label; the bias is the gender.  `datasets/jigsaw_gender/`
2. Biobias dataset: The outcome is the professions; the bias is the gender.  `datasets/biasinbio/`
3. Bold dataset: A generation task where the input is a prompt, and the output is continuous text; the bias is race.  `datasets/bold_new/`

Data format: train/valid/test.json file, each line has an input (column name "document", string), outcome (column name: "label", integer), bias (column name: "gender", integer), datapoint id (column name "annotation_id", integer/string).


Example:
```
{"document": "You act like you never broke the law man, I wish I was as clean as you.", "label": 0, "gender": 0, "annotation_id": 966}
{"document": "'\nTrump is a con man and a buffoon.\n\nNothing more.", "label": 1, "gender": 0, "annotation_id": 2068}
...
```
## Training
### Step 1: Train a single rationale extractor for bias (e.g., gender)

```
CUDA_VISIBLE_DEVICES=$CUDA_ID python main_single.py \
        --mode train \
        --label gender \
        --model latent \
        --save_path $OUT_DIR \
        --resume_snapshot $OUT_DIR/model.pt \ # if you want to resume training
        --batch_size $BS \
        --dependent-z \
        --selection  $SV \
        --lasso $LS \
        --lambda_init $LV 
```

```
$SV: selection rate for rationales
$LS: lasso weight
$LV: lambda value
```
Read the paper to see more explanations for those hyperparameters. The checkpoint will be saved at `$OUT_DIR/model.pt`.



### Step 2: Train a debiased rationale extractor for the outcome (e.g., professions)
1. Load the bias rationale extractor.
2. Train the debiasing rationale extractor; the bias rationale extractor is fixed. 

```
python main_energy.py \
    --bias_model $OUT_DIR/model.pt \
    --save_path $ENERGY_REF_SAVE_DIR
    --eps 1e-5 \
    --num_iterations -100 \
    --batch_size 1024 \
    --sparsity 0.003 \
    --coherence 1.0 \
    --lambda_init 0.001  \
    --word_vectors ./glove.840B.300d.sst.txt \
    --dependent-z \
    --strategy 2 \
    --lambda_min 0.001 \
    --model latent \
    --train_embed \
    --seed $seed \
    --lasso 0.1 \
    --bias_thred $bias_thred \
    --selection $selection \
    --debias_method $debias_method \
    --abs \
    --bias_weight $bias_weight
```

```
num_iterations: number of epochs if less than 0, else number of batches.
word_vectors: path to download word embeddings.
strategy: use 2 to reproduce our work.
dependent-z: if the prediction is dependent on z; use it to reproduce our work.
train_embed: if the embedding layer will be trained.
bias_threshold: what is the bias tolerance threshold in computing the penalty.
abs: if you want to penalize (select - $SV).
bias_weight: bias weight in loss computation.
```
After training the debiased checkpoint is saved to ```$SAVE_DIR```.

## Evaluate
### Prepare a discriminator for classification
We re-use the classifier from the saved single rationale extractor. You can also train your own discriminator as long as it follows the classifier class in `debias_models/common/classifier.py`. 

You can load the discriminator to evaluate the outcome label accuracy (e.g., for professions). In this case, make sure you have trained the corresponding rationale extractor; otherwise, train it now: 

```
CUDA_VISIBLE_DEVICES=$CUDA_ID python main_single.py \
        --mode train \
        --model latent \
        --label label \
        --save_path $DISCRIM_DIR/label \
        --batch_size $BS \
        --dependent-z \
        --selection  $SV \
        --lasso $LS \
        --lambda_init $LV 
```

Or you can evaluate bias prediction accuracy by directly loading the bias discriminator from the bias rationale extractor you saved before:
```
cp $OUT_DIR/model.pt $DISCRIM_DIR/gender/model.pt
```


### Start Evaluating the Debiased Rationale Extractor
1. Load the saved rationale extractor.
2. Load the saved discriminator.
3. Calculate the metrics.


```
CUDA_VISIBLE_DEVICES=0 python main_eval.py --label $LABEL_TYPE \
    --save_path $RESULT_SAVE_TO_DIR \
    --rationale_model_path $ENERGY_REF_SAVE_DIR/model.pt \
    --discrim_model_path $DISCRIM_DIR/$LABEL_TYPE/model.pt \
    --save_predictions 1 \
    --dataset biobias \
    --generate 0 \
    --word_vectors ./glove.840B.300d.sst.txt
```

`$LABEL_TYPE` can be the outcome (e.g., professions) by setting `$LABEL_TYPE=label` or bias (e.g., gender) by setting `$LABEL_TYPE=gender`.


Takeaway:

To reproduce our work in debiasing classification, you do not need to calculate the metrics for generated bias rationale, you only need debiased rationales. Your pipeline is:
1. Train a bias rationale extractor (use bias label).
2. Train an outcome debiased rationale extractor using the energy model(use both bias and outcome label). This model will generate the debiased rationales.
3. Train an outcome discriminator for evaluation (use outcome label).
4. Generate outcome rationales for accuracy, F1, comprehensiveness, and sufficiency evaluation (use outcome label).




