from time import time
import numpy as np
from random import uniform
from sklearn.datasets import make_moons, make_blobs
import matplotlib.pyplot as plt

from engine import Number
from nn import MLP
from graphics import create_frame, create_gif


def MSE(pred_labels: list[Number], true_labels: list[int]) -> Number:
    losses = [(predi - Number(yi)) ** 2 for predi, yi in zip(pred_labels, true_labels)]
    loss = sum(losses) * (1.0 / len(pred_labels))
    return loss


def SVM_loss(pred_labels: list[Number], true_labels: list[int]) -> Number:
    losses = [(1 - Number(yi)*predi).relu() for predi, yi in zip(pred_labels, true_labels)]
    loss = sum(losses) * (1.0 / len(pred_labels))
    return loss


def L2_loss(model: MLP, alpha: float = 1e-4) -> Number:
    reg_loss = alpha * sum((p*p for p in model.parameters()))
    return reg_loss


def accuracy(pred_labels: list[Number], true_labels: list[int]) -> float:
    scores = [(yi > 0) == (predi.value > 0) for predi, yi in zip(pred_labels, true_labels)]
    return sum(scores) / len(scores)


class Trainer():

    def __init__(
            self, 
            model: MLP, 
            train_dataset: list, 
            train_labels: list, 
            eval_dataset: list = None,
            eval_labels: list = None,
            learning_rate: float = 2e-5,
            batch_size: int = None, 
            epochs: int = 1,
            loss_name: str = "MSE",
            reg: str = None,
            plot: str = None,
            gif: str = None,
        ):

        self.model = model

        self.train_set = train_dataset
        self.train_labels = train_labels
        if len(self.train_set) != len(self.train_labels):
            raise ValueError("<train_dataset> and <train_labels> must have the same number of elements")
        
        self.eval_set = eval_dataset
        self.eval_labels = eval_labels
        if self.eval_set is not None and self.eval_labels is not None and len(self.eval_set) != len(self.eval_labels):
            raise ValueError("<eval_dataset> and <eval_labels> must have the same number of elements")

        self.lr = learning_rate

        if batch_size is not None:
            self.batch_size = min(batch_size, len(self.train_set))
        else:
            self.batch_size = len(self.train_set)

        self.epochs = epochs

        self.steps_per_epoch = len(self.train_set) // self.batch_size

        if loss_name == "MSE":
            self.loss_function = MSE
        elif loss_name == "SVM":
            self.loss_function = SVM_loss
        else:
            raise ValueError("Invalid loss function")

        if reg is None:
            self.reg_loss = None
        elif reg.lower() == "l2":
            self.reg_loss = L2_loss
        else:
            self.reg_loss = None
        
        self.plot = plot
        self.gif = gif
    
    def train(self, train_dataset: list = None, train_labels: list = None):

        if train_dataset is not None and train_labels is not None:
            if len(train_dataset) != len(train_labels):
                raise ValueError("<train_dataset> and <train_labels> must have the same number of elements")
            train_set = train_dataset
            train_labels = train_labels
        else:
            train_set = self.train_set
            train_labels = self.train_labels

        train_set_size = len(train_set)
        
        step_loss = []
        step_acc = []

        for epoch in range(self.epochs):

            print(f"\n# EPOCH {epoch+1}/{self.epochs}") if self.steps_per_epoch > 1 else None

            perm = np.random.permutation(train_set_size)

            train_set_epoch = [train_set[idx] for idx in perm]
            train_labels_epoch = [train_labels[idx] for idx in perm]

            for step in range(self.steps_per_epoch):
                
                start = step*self.batch_size
                end = start + self.batch_size

                # forward
                pred = [self.model(input) for input in train_set_epoch[start:end]]

                # loss
                loss = self.loss_function(pred, train_labels_epoch[start:end])
                if self.reg_loss is not None:
                    loss += self.reg_loss(model)
                # print(loss.grad, self.model.parameters()[56])

                # backpropagation
                self.model.zero_grad()
                # print(loss.grad, self.model.parameters()[56])
                loss.backward()
                # print(loss.grad, self.model.parameters()[56])

                # update
                lr = self.lr * (1 - 0.9*epoch/self.epochs)
                for p in self.model.parameters():
                    p.value -= lr * p.grad

                # accuracy
                acc = accuracy(pred, train_labels_epoch[start:end])
                
                print(f"EPOCH {epoch+1} - step {step+1}/{self.steps_per_epoch} | loss = {loss.value:.8f} | accuracy = {100*acc:.2f}%")
                step_loss.append(loss.value)
                step_acc.append(acc)
            
            if self.eval_set is not None and self.eval_labels is not None:
                print("NO EVAL for the moment...")

            if self.gif is not None:
                create_frame(
                    t=epoch,
                    model=self.model,
                    train_dataset=train_set,
                    train_labels=train_labels,
                    gif_name=self.gif,
                )
        
        if self.plot is not None:
            steps = [k for k in range(self.steps_per_epoch * self.epochs)]
            plt.plot(steps, step_loss, color="blue", label="loss", marker="o")
            plt.plot(steps, step_acc, color="orange", label="accuracy", marker="o")
            plt.legend()
            plt.xlabel("steps")
            plt.ylabel("value")
            plt.title(f"Training of MLP{self.model.size} for {self.epochs} epochs on {len(train_set)} samples")
            plt.savefig(self.plot)
        
        if self.gif is not None:
            create_gif(self.gif, self.epochs)
                

if __name__ == "__main__":
    
    size = [2, 16, 16, 1]
    model = MLP(size, bias=True)
    n = 100

    moons_x, moons_y = make_moons(n_samples=n, noise=0.1)
    moons_y = moons_y*2 - 1 # map 1 -> 1 and 0 -> -1
    train_dataset = moons_x.tolist()
    train_labels = moons_y.tolist()
    # print(train_dataset[:3])
    # print(train_labels[:3])

    # visualize in 2D
    if False:
        plt.figure(figsize=(5,5))
        plt.scatter(moons_x[:,0], moons_x[:,1], c=moons_y, s=20, cmap='jet')
        plt.savefig("dataset.png")

    trainer = Trainer(
        model=model, 
        train_dataset=train_dataset, 
        train_labels=train_labels, 
        eval_dataset=None,
        eval_labels=None,
        learning_rate=1.0,
        batch_size=int(float(n)/1), 
        epochs=100,
        loss_name="SVM",
        reg="L2",
        plot="training_SVM_L2.png",
    )
    
    trainer.train()


