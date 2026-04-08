# Environment Setting

```bash
# conda
conda create -n lbm python=3.12
conda activate lbm
conda env list
conda deactivate
conda remove -n lbm --all

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
# run policy server (example)
python /data/maxshen/lbm_eval/grpc_workspace/wave_around_policy_server.py
## Apple 5k steps test
python /data/maxshen/lbm_eval/grpc_workspace/lerobot_policy_server.py --model_id=/data/maxshen/lerobot/outputs/train/2026-04-07/15-05-11_diffusion/checkpoints/last/pretrained_model

## Apple 150k steps
python /data/maxshen/lbm_eval/grpc_workspace/lerobot_policy_server.py --model_id=/data/maxshen/lerobot/outputs/train/2026-04-07/14-07-57_diffusion/checkpoints/last/pretrained_model --server-uri=localhost:51212

## Banana 150k steps
python /data/maxshen/lbm_eval/grpc_workspace/lerobot_policy_server.py --model_id=/data/maxshen/lerobot/outputs/train/2026-04-07/01-30-38_diffusion/checkpoints/last/pretrained_model --server-uri=localhost:51212

## Kiwi 150k steps
python /data/maxshen/lbm_eval/grpc_workspace/lerobot_policy_server.py --model_id=/data/maxshen/lerobot/outputs/train/2026-04-07/01-28-57_diffusion/checkpoints/last/pretrained_model --server-uri=localhost:51212

# run env clinet server (example)
evaluate --skill_type=pick_and_place_box --num_evaluations=1 --num_processes=1 --output_dir=output
# apple
evaluate --skill_type=bimanual_place_apple_from_bowl_on_cutting_board --num_evaluations=1 --num_processes=1 --output_dir=output --server_uri=localhost:51212
# banana
evaluate --skill_type=put_banana_on_saucer --num_evaluations=1 --num_processes=1 --output_dir=output --server_uri=localhost:51212
# kiwi
evaluate --skill_type=put_kiwi_in_center_of_table --num_evaluations=1 --num_processes=1 --output_dir=output --server_uri=localhost:51212

```
