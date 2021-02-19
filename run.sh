#!/bin/bash
project="bat_test"
gpu="p100,t4v2,rtx6000"
enable_wandb=true #true/false
epoch=100
#bat_k=3
#arch="resnet34"
#arch="wideresnet"
lr_update="multistep"
lr=0.1

for bat_k in 1 3; do
	#for arch in "resnet8" "resnet34"; do
	for arch in "resnet8_mm" "resnet34_mm"; do
	#for arch in "resnet8" "resnet34" "wideresnet"; do
		for bat_step in 100 300 500 1000; do
		#for bat_step in 100 200; do
		#for bat_step in 300 500; do
			# for method in "pgd" "standard" "fgsm" "bat_fgsm"; do
			# for method in "pgd" "standard" "fgsm"; do
			# for method in "pgd" "fgsm"; do
			for method in "bat_pgd" "bat_exp_pgd"; do
			#for method in "bat_exp_pgd"; do
			#for method in "bat_exp_fgsm"; do
				bash launch_slurm_job.sh ${gpu} ${arch}_${method}_k${bat_k}_step${bat_step} 1 "python3 main.py --method \"${method}\" --arch \"${arch}\" --lr_update \"${lr_update}\" --bat_k ${bat_k} --bat_step ${bat_step} --lr ${lr} --dataset \"cifar10\" --epoch ${epoch} --wandb_project \"${project}\" --enable_wandb ${enable_wandb}"
			done
		done
	done
done

#for arch in "resnet8_mm" "resnet34_mm"; do
	##for method in "pgd_BN" "pgd"; do
	#for method in "pgd"; do
		#bash launch_slurm_job.sh ${gpu} ${arch}_${method} 1 "python3 main.py --method \"${method}\" --arch \"${arch}\" --lr_update \"${lr_update}\" --lr ${lr} --dataset \"cifar10\" --epoch ${epoch} --wandb_project \"${project}\" --enable_wandb ${enable_wandb}"
	#done
#done

##for arch in "resnet8"; do
## for arch in "c11"; do
## for arch in "resnet34" "wideresnet"; do
	#for method in "ensemble_pgd_standard"; do
	## for method in "ensemble_pgd_fgsm" "ensemble_pgd_trades" "ensemble_pgd_optim_v1" "ensemble_pgd_optim_v2"; do
		#bash launch_slurm_job.sh ${gpu} ${arch}_${method} 1 "python3 main.py --method \"${method}\" --arch \"${arch}\" --lr_update \"${lr_update}\"  --lr ${lr} --dataset \"cifar10\" --epoch ${epoch} --wandb_project \"${project}\" --enable_wandb ${enable_wandb}"
	#done
#done
