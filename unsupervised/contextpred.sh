CUDA_LAUNCH_BLOCKING=1 \
OMP_NUM_THREADS=1 \
CUDA_VISIBLE_DEVICES=0,1,2,3 \
torchrun \
--nproc_per_node=4 \
--nnodes=1          \
--node_rank=0       \
--master_addr=localhost  \
--master_port=6006 \
pretrain_contextpred.py