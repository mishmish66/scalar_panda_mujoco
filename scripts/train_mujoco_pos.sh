#!/bin/bash
#SBATCH -n 1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH --time=0-10:00:00
#SBATCH --partition=cpu
#SBATCH --output=/home/wschwarzer_umass_edu/work/logs/train_mujoco_%j.out
#SBATCH --error=/home/wschwarzer_umass_edu/work/logs/train_mujoco_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=wschwarzer@umass.edu

# conda activate rings
python agent_training.py --algorithm PPO --gamma 0.99 --no-use-touch-rewards --no-use-shaping-rewards --destination-pos 0.5,0.0,0.5 --max-steps 100 --learning-steps 100000 --center-rewards --test-steps 1000 --test-render-interval 10