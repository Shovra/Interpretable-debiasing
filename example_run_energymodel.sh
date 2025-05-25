while getopts d:s:t:l:w: flag
do
    case "${flag}" in
        d) debias_method=${OPTARG};;
        s) seed=${OPTARG};;
        t) bias_thred=${OPTARG};;
        l) selection=${OPTARG};;
        w) bias_weight=${OPTARG};;
    esac
done
echo "debias_method: $debias_method";
echo "seed: $seed";
echo "bias_thred: $bias_thred";
echo "selection: $selection";
echo "bias_weight: $bias_weight";
echo kuma_${debias_method}_embed_eps1e-5_seed${seed}_thred${bias_thred}_selection${selection}_abs_baseOn5_BIAS${bias_weight}.out

python main_energy.py --bias_model /data/wangyu/biobias_results/gender/gender_kuma_strategy2_abs_embed_selection0.5_seed0/gender_model.pt \
             --eps 1e-5 --num_iterations -100 --batch_size 1024 --sparsity 0.0003 --coherence 1.0 --lambda_init 0.001  \
             --word_vectors /data/wangyu/glove.840B.300d.sst.txt --dependent-z --strategy 2 --lambda_min 0.001 --model latent \
             --train_embed --seed $seed --lasso 0.0 --bias_thred $bias_thred --selection $selection --debias_method $debias_method --abs \
             --bias_weight $bias_weight &> kuma_${debias_method}_embed_eps1e-5_seed${seed}_thred${bias_thred}_selection${selection}_abs_baseOn5_BIAS${bias_weight}.out

