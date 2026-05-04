# Environment Setting

```bash
# conda
conda create -n lbm python=3.12
conda activate lbm
conda env list
conda deactivate
conda remove -n lbm --all
pip install lerobot
# tmux
tmux new -s window_name
crrl + b, d
tmux kill-session -t window_name
tmux ls
tmux a -t window_name

# utilities
rsync -av --progress a b
rsync -av --progress /data/kylehatch/LBM_sim_egocentric /data/maxshen/Video_data2
watch -n 1 nvidia-smi
export CUDA_VISIBLE_DEVICES=x
echo $CUDA_VISIBLE_DEVICES

# lbm
cd lbm_eval
conda activate lbm

# run policy server

# check if the port is occupied
ss -ltnp | grep ':51211 '
lsof -iTCP:51211 -sTCP:LISTEN -n -P

ss -ltnp | grep ':51212 '
lsof -iTCP:51212 -sTCP:LISTEN -n -P

ss -ltnp | grep ':51213 '
lsof -iTCP:51213 -sTCP:LISTEN -n -P

ss -ltnp | grep ':51214 '
lsof -iTCP:51214 -sTCP:LISTEN -n -P

## Multi-camera policy server
python -m grpc_workspace.lerobot_policy_server \
--model_id=/data/maxshen/lerobot/outputs/train/2026-04-28/01-52-40_diffusion/checkpoints/045000/pretrained_model --server-uri=localhost:51211 --cameras scene_right_0 wrist_right_minus wrist_left_plus > outputs/egoAndWrist/apple_policy_server.log 2>&1 &

# Single-camera policy server
python -m grpc_workspace.lerobot_policy_server \
--model_id=/data/maxshen/lerobot_checkpoints/apple_egoAndWrist/015000/pretrained_model --server-uri=localhost:51212 > outputs/apple_policy_server.log 2>&1 &

python -m grpc_workspace.lerobot_policy_server \
--model_id=/data/maxshen/lerobot_checkpoints/banana_only_ego/100000/pretrained_model --server-uri=localhost:51213 > outputs/banana_policy_server.log 2>&1 &

python  -m grpc_workspace.lerobot_policy_server \
--model_id=/data/maxshen/lerobot_checkpoints/kiwi_only_ego/100000/pretrained_model --server-uri=localhost:51214 > outputs/kiwi_policy_server.log 2>&1 &

python -m grpc_workspace.dit_policy_server \
--model_id=/data/maxshen/lerobot/outputs/train/2026-05-03/01-45-44_multitask-dit/checkpoints/040000/pretrained_model \
--server-uri=localhost:51211 \
--task_name=BimanualPlaceAppleFromBowlOnCuttingBoard \
--cameras scene_right_0 wrist_right_minus wrist_left_plus > outputs/egoAndWrist/apple_policy_server.log 2>&1 &

# run env clinet server
## apple
python -m lbm_eval.evaluate \
--skill_type=bimanual_place_apple_from_bowl_on_cutting_board \
--num_evaluations=20 \
--num_processes=2 \
--output_dir=outputs/egoAndWrist \
--server_uri=localhost:51211

python -m lbm_eval.evaluate \
--skill_type=bimanual_place_apple_from_bowl_on_cutting_board \
--num_evaluations=20 \
--num_processes=2 \
--output_dir=outputs2 \
--server_uri=localhost:51212

## banana
python -m lbm_eval.evaluate \
--skill_type=put_banana_on_saucer \
--num_evaluations=20 \
--num_processes=2 \
--output_dir=outputs \
--server_uri=localhost:51213

## kiwi
python -m lbm_eval.evaluate \
--skill_type=put_kiwi_in_center_of_table \
--num_evaluations=20 \
--num_processes=2 \
--output_dir=outputs \
--server_uri=localhost:51214
```
