import numpy as np

drawings = np.load("data/cat.npy", mmap_mode="r")
sample = drawings[:5000]

drawing = sample[0].reshape(28, 28)
for row in drawing:
    print("".join("██" if pixel >= 72 else "  " for pixel in row))