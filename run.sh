#!/bin/bash
project="bat_test"
gpu="t4v2"
enable_wandb=true #true/false
epoch=100
bat_k=4
arch="resnet8"
lr_update="multistep"
lr=0.1

for bat_step in 1; do
	# for method in "pgd" "standard" "fgsm" "bat_fgsm"; do
	# for method in "pgd" "standard" "fgsm"; do
	# for method in "pgd" "fgsm"; do
	for method in "bat_fgsm" "bat_pgd"; do
		bash launch_slurm_job.sh ${gpu} ${arch}_${method}_${bat_step} 1 "python3 main.py --method \"${method}\" --arch \"${arch}\" --lr_update \"${lr_update}\" --bat_k ${bat_k} --bat_step ${bat_step} --lr ${lr} --dataset \"cifar10\" --epoch ${epoch} --wandb_project \"${project}\" --enable_wandb ${enable_wandb}"
	done
done