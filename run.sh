#!/bin/bash
project="bat_test"
gpu="t4v2"
enable_wandb=true #true/false
epoch=10

for lr in 0.01; do
	# for method in "pgd" "standard"; do
	# for method in "pgd"; do
	for method in "bat_fgsm"; do
		bash launch_slurm_job.sh ${gpu} jobname_${lr}_${method} 1 "python3 main.py --method \"${method}\" --lr ${lr} --dataset \"cifar10\" --epoch ${epoch} --wandb_project \"${project}\" --enable_wandb ${enable_wandb}"
	done
done