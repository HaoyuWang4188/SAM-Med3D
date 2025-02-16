# python train.py --multi_gpu
# CUDA_VISIBLE_DEVICES=1 python train.py --batch_size 6 --task_name "test_segFM" ---lr 8e-5 
# CUDA_VISIBLE_DEVICES=0 python train.py --batch_size 6 --task_name "base_segFM" --checkpoint "" #--lr 8e-5 
# CUDA_VISIBLE_DEVICES=0,1 python train.py --batch_size 6 --multi_gpu --task_name "ft_segFM" --lr 8e-5 
# CUDA_VISIBLE_DEVICES=0,1 python train_bbox.py --batch_size 6 --multi_gpu --task_name "test_bbox_ft_segFM" --lr 8e-5 
CUDA_VISIBLE_DEVICES=0,1 python train_bbox.py --batch_size 6 --multi_gpu --task_name "test_bbox_ft_segFM" --lr 8e-5 
