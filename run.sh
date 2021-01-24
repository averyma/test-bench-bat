#!/bin/bash
project="bat_test"
gpu="t4v2"
enable_wandb=true #true/false
epoch=100
bat_k=2
# arch="resnet34"
# arch="wideresnet"
lr_update="multistep"
lr=0.1

for arch in "resnet8" "resnet34" "wideresnet"; do
	for bat_step in 300 500 1000; do
		# for method in "pgd" "standard" "fgsm" "bat_fgsm"; do
		# for method in "pgd" "standard" "fgsm"; do
		# for method in "pgd" "fgsm"; do
		for method in "bat_pgd"; do
			bash launch_slurm_job.sh ${gpu} ${arch}_${method}_k${bat_k}_step${bat_step} 1 "python3 main.py --method \"${method}\" --arch \"${arch}\" --lr_update \"${lr_update}\" --bat_k ${bat_k} --bat_step ${bat_step} --lr ${lr} --dataset \"cifar10\" --epoch ${epoch} --wandb_project \"${project}\" --enable_wandb ${enable_wandb}"
		done
	done
done

# for arch in "resnet8" "resnet34" "wideresnet"; do
# 	for method in "pgd"; do
# 		bash launch_slurm_job.sh ${gpu} ${arch}_${method} 1 "python3 main.py --method \"${method}\" --arch \"${arch}\" --lr_update \"${lr_update}\" --lr ${lr} --dataset \"cifar10\" --epoch ${epoch} --wandb_project \"${project}\" --enable_wandb ${enable_wandb}"
# 	done
# done

# for arch in "resnet8" "resnet34" "wideresnet"; do
# 	for method in "ensemble_pgd"; do
# 		bash launch_slurm_job.sh ${gpu} ${arch}_${method} 1 "python3 main.py --method \"${method}\" --arch \"${arch}\" --lr_update \"${lr_update}\"  --lr ${lr} --dataset \"cifar10\" --epoch ${epoch} --wandb_project \"${project}\" --enable_wandb ${enable_wandb}"
# 	done
# done