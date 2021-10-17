env="MPE"
scenario="formation_hd_env"
num_agents=3
algo="rmappo"
exp="render"
seed_max=1

echo "env is ${env}"
for seed in `seq ${seed_max}`
do
    # CUDA_VISIBLE_DEVICES=1 python render_formation.py --save_gifs --share_policy --env_name ${env} --algorithm_name ${algo} --experiment_name ${exp} --scenario_name ${scenario} --num_agents ${num_agents} --seed ${seed} --n_training_threads 1 --n_rollout_threads 1 --use_render --episode_length 25 --render_episodes 5 --model_dir "./results/MPE/formation_hd_env/rmappo/check/run15/models"
    python render_formation.py --save_gifs --env_name ${env} --algorithm_name ${algo} --experiment_name ${exp} --scenario_name ${scenario} --num_agents ${num_agents} --seed ${seed} --n_training_threads 1 --n_rollout_threads 1 --use_render --episode_length 50 --render_episodes 5 --model_dir "/Users/reedpan/Desktop/Research/gym_formation/train/mappo/results/MPE/formation_hd_env/rmappo/hd_3/run1/models" --gif_dir './results/gif'
done