#!/bin/bash
#SBATCH --job-name=fm_run_utk_face_down      # Job name
#SBATCH --output=logs/utk_face/out/out_fm_utk_face_down_per_epoch_continue.txt   # Standard output log file
#SBATCH --error=logs/utk_face/err/err_fm_run_utk_face_down_continue.txt    # Standard error log file
#SBATCH --partition=a6000   # Specify the partition (or queue) if needed
#SBATCH --gres=gpu:a6000:1  # Request one A6000 GPU
#SBATCH --time=24:00:00     # Set the maximum runtime (adjust as needed)
#SBATCH --nodes=1           # Request one node
#SBATCH --ntasks=1          # Number of tasks (for single-GPU job, usually 1)
#SBATCH --cpus-per-task=8   # Number of CPU cores per task (adjust based on your needs)

# Load required modules or environment (adjust as necessary)
module load python/3.8
module load cuda/11.0  # Adjust based on available CUDA version

# Run the Python script
srun python3.8 main_feature_memorization2.py --dataset_name utk_face --model_name resnet18 --desired_noise_label 3 --desired_test_label 2 --batch_size 512 --down True --device cuda:0