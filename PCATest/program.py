from sklearn.datasets import fetch_olivetti_faces
from scipy import linalg as la
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimage 
import math

def cov(X):
    print("calculating covariance matrix...")
    mult = np.dot(X.T, X)
    return mult / X.shape[0]

def PCA (data, d = 100):
    print("data size:")
    print(data.shape)
    mean = data.mean(axis=0)
    data = data - mean
    R = cov(data)

    print("calculating eigen vectors and values...")
    evals, evecs = la.eigh(R)
    idx = np.argsort(evals)[::-1]
    evecs = evecs[:, idx]
    evals = evals[idx]

    print("transforming to new space...")
    evecs = evecs[:, :d]
    trans = np.dot(data, evecs)
    recon = np.dot(trans, evecs.T) + mean

    return trans, recon

def Plot(data, recon, rows=5, cols=2):
    fig = plt.figure()    
    plt.set_cmap('gray')

    for row in range(0, rows):
        for col in range(0, cols):
            idx = np.random.randint(0, data.shape[0])
            img_before = np.reshape(data[idx], (64, 64))
            img_after = np.reshape(recon[idx], (64, 64))
            idx = row * cols * 2 + col * 2
            fig.add_subplot(rows, cols*2, idx + 1)
            plt.axis('Off')
            plt.imshow(img_before)
            fig.add_subplot(rows, cols*2, idx + 2)
            plt.axis('Off')
            plt.imshow(img_after)

    plt.show()

"""
# test data
data = np.array([np.random.randn(8) for k in range(50)])
data[:50, 2:4] += 5
data[50:, 2:5] += 5

# visualize
trans, recon = PCA(data, 7)
fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
ax1.scatter(data[:50, 0], data[:50, 1], c = 'r')
ax1.scatter(data[50:, 0], data[50:, 1], c = 'b')
ax2.scatter(trans[:50, 0], trans[:50, 1], c = 'r')
ax2.scatter(trans[50:, 0], trans[50:, 1], c = 'b')
ax3.scatter(recon[:50, 0], recon[:50, 1], c = 'r')
ax3.scatter(recon[50:, 0], recon[50:, 1], c = 'b')
plt.show()
"""

dataset = fetch_olivetti_faces(shuffle=True)
data, labels = dataset.data, dataset.target

trans, recon = PCA(data, 20*20)

Plot(data, recon, 8, 4)
