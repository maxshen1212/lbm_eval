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

## Apple 150k steps
python3 -m grpc_workspace.lerobot_policy_server \
--model_id=/data/maxshen/lerobot/outputs/train/2026-04-11/05-05-58_diffusion/checkpoints/050000/pretrained_model --server-uri=localhost:51212 > outputs/apple_policy_server.log 2>&1 &

## Banana 150k steps
python3 -m grpc_workspace.lerobot_policy_server \
--model_id=/data/maxshen/lerobot/outputs/train/2026-04-11/05-06-09_diffusion/checkpoints/last/pretrained_model --server-uri=localhost:51213 > outputs/banana_policy_server.log 2>&1 &

## Kiwi 150k steps
python3 -m grpc_workspace.lerobot_policy_server \
--model_id=/data/maxshen/lerobot/outputs/train/2026-04-11/05-06-17_diffusion/checkpoints/last/pretrained_model --server-uri=localhost:51214 > outputs/kiwi_policy_server.log 2>&1 &

# run env clinet server
## apple
python3 -m lbm_eval.evaluate \
--skill_type=bimanual_place_apple_from_bowl_on_cutting_board \
--num_evaluations=20 \
--num_processes=3 \
--output_dir=outputs2 \
--server_uri=localhost:51212

## banana
python3 -m lbm_eval.evaluate \
--skill_type=put_banana_on_saucer \
--num_evaluations=20 \
--num_processes=3 \
--output_dir=outputs \
--server_uri=localhost:51213

## kiwi
python3 -m lbm_eval.evaluate \
--skill_type=put_kiwi_in_center_of_table \
--num_evaluations=20 \
--num_processes=3 \
--output_dir=outputs \
--server_uri=localhost:51214
```
