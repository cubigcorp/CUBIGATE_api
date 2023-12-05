import numpy as np
import argparse


def cosine(a: np.ndarray, b: np.ndarray):
    dot = np.dot(a, b)
    norm = np.linalg.norm(a) * np.linalg.norm(b)
    return dot / norm

parser = argparse.ArgumentParser()
parser.add_argument(
    '--log',
    type=str,
    required=True
)
args = parser.parse_args()
npz = np.load(args.log)
public_features = npz['public_features']
distances = npz['distances']
ids = npz['ids']
count = npz['count']
npz.close()

num = public_features.shape[0]
sim = 0.
idx = 0
for i in range(num):
    for j in range(i + 1, num):
        sim += cosine(public_features[i], public_features[j])
        idx += 1
sim /= idx
print(f"Average cosine similarity: {sim}")
distances = distances.flatten()
print(f"Average distance: {np.average(distances)}")
print(f"distance variance: {np.var(distances)}")