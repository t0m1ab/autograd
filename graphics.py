import os

import numpy as np
import matplotlib.pyplot as plt
import imageio

from engine import Number
from nn import MLP


def create_frame(t: int, model: MLP, train_dataset: list[list[float]], train_labels: list[float], gif_name: str):

    gif_path = os.path.join("./output", gif_name)
    if not os.path.isdir(gif_path):
        os.mkdir(gif_path)
        print(f"Create directory '{gif_path}'")

    frames_path = os.path.join(gif_path, "frames/")
    if not os.path.isdir(frames_path):
        os.mkdir(frames_path)
        print(f"Create directory '{frames_path}'")
    
        
    h = 0.25
    np_train_dataset = np.array(train_dataset)
    x_min, x_max = np_train_dataset[:, 0].min() - 1, np_train_dataset[:, 0].max() + 1
    y_min, y_max = np_train_dataset[:, 1].min() - 1, np_train_dataset[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Xmesh = np.c_[xx.ravel(), yy.ravel()]
    inputs = [list(map(Number, xrow)) for xrow in Xmesh]
    scores = list(map(model, inputs))
    Z = np.array([s.value > 0 for s in scores])
    Z = Z.reshape(xx.shape)

    fig = plt.figure()
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral, alpha=0.8)
    plt.scatter(np_train_dataset[:, 0], np_train_dataset[:, 1], c=train_labels, s=40, cmap=plt.cm.Spectral)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.title(f"EPOCH {t+1}")

    plt.savefig(os.path.join(frames_path, f"frame_{t}.png"), transparent = False, facecolor = 'white')

    plt.close()


def create_gif(gif_name: str, epochs: int):

    gif_path = os.path.join("./output", gif_name)
    if not os.path.isdir(gif_path):
        raise IsADirectoryError(f"Unable to find directory: '{gif_path}'")
    
    frame_names = []
    for t in range(epochs):
        frame_name = os.path.join(gif_path, "frames/", f"frame_{t}.png")
        if not os.path.isfile(frame_name):
            raise FileExistsError(f"Unable to find file: '{frame_name}'")
        frame_names.append(frame_name)
    
    frames = [imageio.v2.imread(x) for x in frame_names]

    gif_filepath = os.path.join(gif_path, f"{gif_name}.gif")
    imageio.mimsave(
        gif_filepath,
        frames,
        duration=500, # ms/frame
    )

    print(f"GIF saved at: {gif_filepath}")