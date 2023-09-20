"""
Convert .mat files keys to fit to FUnmix notations
"""
import scipy.io as sio
import os

# NOTE: Modify the line below manually
name = "MixedRatio_rho1.0"
path = os.path.join("data", f"{name}.mat")
data = sio.loadmat(path)

print(f"Old keys: {data.keys()}")

# Convert keys
mapping = {
    "Y": "Y",
    "E": "E",
    "A": "A",
    "D": "D",
    "H": "h",
    "W": "w",
    "p": "r",
    "L": "p",
    "N": "n",
    "M": "m",
    "index": "index",
    "labels": "labels",
}

new_data = {mapping[k]: data[k] for k in data if not k.startswith("__")}
print(f"New keys: {new_data.keys()}")

sio.savemat(path, new_data)
