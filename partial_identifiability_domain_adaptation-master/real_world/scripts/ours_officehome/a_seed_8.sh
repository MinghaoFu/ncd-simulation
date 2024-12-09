declare -a s_dim=(4)
declare -a seed=(8)
declare -a lambda_vae=(1e-4)
declare -a gpu=(1)

for ((i=0;i<${#seed[@]};++i)); do
for ((j=0;j<${#s_dim[@]};++j)); do
for ((k=0;k<${#lambda_vae};++k})); do
WANDB_MODE=disabled CUDA_VISIBLE_DEVICES=${gpu[0]} python main.py --root=../../da_datasets/office-home --batch-size=96 --train_batch_size=32 \
-d OfficeHome -s Cl,Pr,Rw -t Ar -a resnet50 \
--name=officehome_search \
--z_dim=128 --s_dim=${s_dim[j]}  \
--C_max=35 --beta=1 --lambda_vae=${lambda_vae[k]} --lambda_ent=0.1 \
-i=1000 \
--seed=${seed[i]}
done
done
done