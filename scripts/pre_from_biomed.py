import torchio as tio
from glob import glob
import os
import os.path as osp
from scipy.ndimage import zoom
import numpy as np

import multiprocessing as mp
from functools import partial
from tqdm import tqdm

def read_data_from_npz(npz_file):
    data = np.load(npz_file)
    # print(data["spacing"])
    return data['imgs'], data['gts'], data["spacing"]

def pad_and_resize(array, target_shape=[128, 128, 128]):
    pad_width = []
    for i in range(3):
        total_pad = max(0, target_shape[i] - array.shape[i])
        pad_before = total_pad // 2
        pad_after = total_pad - pad_before
        pad_width.append((pad_before, pad_after))
    padded_array = np.pad(array, pad_width, mode='constant', constant_values=0)

    zoom_factors = [target_shape[i] / padded_array.shape[i] for i in range(3)]
    resized_array = zoom(padded_array, zoom_factors, order=1)  # order=1 for bilinear interpolation
    return resized_array, zoom_factors

def preprocess(npz_path, output_dir):
    fname = osp.basename(npz_path).replace(".npz", "")
    dataset = osp.basename(osp.dirname(npz_path))
    modality = osp.basename(osp.dirname(osp.dirname(npz_path)))
    out_dir = osp.join(output_dir, modality, dataset)
    os.makedirs(out_dir, exist_ok=True)
    # print(fname, "->", out_path)

    # start to preprocess data
    img, gt, spacing = read_data_from_npz(npz_path)
    # print("orig:", img.shape, gt.shape, spacing)
    img, _ = pad_and_resize(img)
    gt, factors = pad_and_resize(gt)
    scales = [factors[2], factors[0], factors[1]]
    spacing = [s / sc for s, sc in zip(spacing, scales)]
    # print("curr:", img.shape, gt.shape, spacing)

    for idx, cls_idx in enumerate(np.unique(gt)):
        out_path = osp.join(out_dir, f"{fname}_cls{cls_idx}.npz")
        cls_gt = np.zeros_like(gt)
        if cls_idx == 0:
            continue
        cls_gt[gt == cls_idx] = 1
        np.savez(out_path, img=img, gt=cls_gt, spacing=spacing)






if __name__ == "__main__":
    dataet_dir = "./biomed_baseline/3D_train_npz_random_10percent_16G"
    output_dir = "./biomed_baseline_pre/3D_train_npz_random_10percent_16G"
    os.makedirs(output_dir, exist_ok=True)

    all_npz_path = glob(osp.join(dataet_dir, "*", "*", "*.npz"))#[:20]

    num_workers=32
    preprocess_tr = partial(preprocess, output_dir=output_dir)
    with mp.Pool(num_workers) as p:
        with tqdm(total=len(all_npz_path)) as pbar:
            pbar.set_description("Preprocessing training data")
            for i, _ in tqdm(enumerate(p.imap_unordered(preprocess_tr, all_npz_path))):
                pbar.update()

