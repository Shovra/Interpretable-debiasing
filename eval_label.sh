python main_eval.py --label label \
    --save_path /data2/zexue/debias_by_rationale/jigsaw_results/evaluate/debias/outcome/kuma_strategy2_abs_eps1e-6_seed0/ \
    --rationale_model_path /data2/zexue/debias_by_rationale/jigsaw_results/debias/Energy/kuma_strategy2_abs_eps1e-6_seed0/model.pt \
    --discrim_model_path /data2/zexue/debias_by_rationale/jigsaw_results/discrim/label/model.pt \
    --word_vectors /data2/zexue/interpretable_predictions/data/gender/glove.840B.300d.sst.txt

python main_eval.py --label label \
    --save_path /data2/zexue/debias_by_rationale/jigsaw_results/evaluate/debias/outcome/kuma_strategy2_abs_eps1e-5_seed0/ \
    --rationale_model_path /data2/zexue/debias_by_rationale/jigsaw_results/debias/Energy/kuma_strategy2_abs_eps1e-5_seed0/model.pt \
    --discrim_model_path /data2/zexue/debias_by_rationale/jigsaw_results/discrim/label/model.pt \
    --word_vectors /data2/zexue/interpretable_predictions/data/gender/glove.840B.300d.sst.txt
