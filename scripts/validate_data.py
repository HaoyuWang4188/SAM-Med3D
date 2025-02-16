import numpy as np

def read_data_from_npz(npz_file):
    data = np.load(npz_file)
    print(data["spacing"])
    return data['imgs'], data['gts'], data["spacing"]


if __name__ == "__main__":
    filename = "./biomed_baseline/3D_train_npz_random_10percent_16G/Microscopy/cremi/cremi_000_sc.npz"
    img, gt, spacing = read_data_from_npz(filename)
    print(np.unique(gt))