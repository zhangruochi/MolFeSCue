for i in {0..0}
do
    WORLD_SIZE=0 CUDA_VISIBLE_DEVICES=0 torchrun \
                --nproc_per_node=1 \
                --nnodes=1          \
                --node_rank=0       \
                --master_addr=localhost  \
                --master_port=22223 \
                train_fewshot.py "train.random_seed=$i"
done