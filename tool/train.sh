python train.py  --gpu=0 --net=DisenNet --batch_size=1 \
    --valid_freq=5 --output_set=train_val \
    --DisenNet_indim=8 --AuxDec_dim=4 \
    --miss_modal=True --use_Bernoulli_train=True \
    --use_contrast=True --use_freq_contrast=True \
    --use_distill=True --use_kd=True --affinity_kd=True \
    --setting=D2Net 
