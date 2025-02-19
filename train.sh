# python train.py --multi_gpu
# CUDA_VISIBLE_DEVICES=1 python train.py --batch_size 6 --task_name "test_segFM" ---lr 8e-5 
# CUDA_VISIBLE_DEVICES=0 python train.py --batch_size 6 --task_name "base_segFM" --checkpoint "" #--lr 8e-5 
# CUDA_VISIBLE_DEVICES=0,1 python train.py --batch_size 6 --multi_gpu --task_name "ft_segFM" --lr 8e-5 
# CUDA_VISIBLE_DEVICES=0,1 python train_bbox.py --batch_size 6 --multi_gpu --task_name "test_bbox_ft_segFM" --lr 8e-5 
# CUDA_VISIBLE_DEVICES=0 python train_bbox.py --batch_size 6 --task_name "test_ft_segFM3D_pre" --lr 8e-5 
# CUDA_VISIBLE_DEVICES=0,1 python train_bbox.py --batch_size 6 --multi_gpu --task_name "ft_FM3D_pre_b6x2" --lr 8e-5 --gpu_ids 0 1 
# CUDA_VISIBLE_DEVICES=1 python train_bbox_dec.py --batch_size 6 --task_name "ft_dec_b6x1" --checkpoint "/mnt/sh_flex_storage/home/wanghaoy/code/SAM_Med3D_debug/SAM-Med3D/work_dir/ft_FM3D_pre_b6x1/sam_model_latest.pth" --lr 8e-5 --gpu_ids 0
CUDA_VISIBLE_DEVICES=0,1 python train_bbox_dec.py --batch_size 6 --task_name "ft_dec_b6x2" --checkpoint "/mnt/sh_flex_storage/home/wanghaoy/code/SAM_Med3D_debug/SAM-Med3D/work_dir/ft_FM3D_pre_b6x1/sam_model_latest.pth" --lr 8e-5 --multi_gpu --gpu_ids 0 1
