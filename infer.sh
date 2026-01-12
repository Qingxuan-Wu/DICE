python setup.py build develop
exp_name="example_inference"
mkdir output
mkdir output/$exp_name/
mkdir output/$exp_name/vis_frames/
export WANDB_MODE="dryrun"
cp src/tools/infer.py output/$exp_name/
cp src/modeling/bert/model.py output/$exp_name/
CUDA_VISIBLE_DEVICES=0 \
python -m torch.distributed.launch --nproc_per_node=1 --master_port 47788  \
       src/tools/infer.py \
       --arch hrnet-w64 \
       --num_workers 24 \
       --per_gpu_train_batch_size 1 \
       --per_gpu_eval_batch_size 1 \
       --output_dir output/$exp_name/ \
       --run_name $exp_name \
       --input_feat_dim "2051,512,128" \
       --hidden_feat_dim "1024,256,128" \
       --output_feat_dim "512,128,3" \
       --input_folder assets/images/ \
       --resume_checkpoint checkpoints/model.bin \
       --num_train_epochs 1
