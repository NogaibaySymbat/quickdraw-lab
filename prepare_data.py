import numpy as np

classes = ["cat", "fish", "airplane"]

for label, name in enumerate(classes):
    drawings = np.load(f"data/{name}.npy", mmap_mode="r")
    sample = drawings[:5000]
    print(label, name, sample.shape)