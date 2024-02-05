import threading
import torch
from torch.utils.data import DataLoader

from server.mnist import model, loss_fn, train_data


class JobThread(threading.Thread):
    def __init__(self, job_id, batch_size, learning_rate, epochs, progress_callback):
        super().__init__()
        self.job_id = job_id

        self.train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
        self.optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
        self.epochs = epochs

        self.progress_callback = progress_callback
        self.progress = 0
    

    def run(self):
        total_batches = len(self.train_dataloader)
        batches_per_epoch = total_batches / self.epochs

        for epoch in range(self.epochs):
            print(f'Training epoch {epoch + 1}...')

            for batch, (X,y) in enumerate(self.train_dataloader):
                # compute prediction and loss
                pred = model(X)
                loss = loss_fn(pred, y)

                # backpropagation
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()

                # calculate progress
                progress = (epoch * batches_per_epoch + batch + 1) / (batches_per_epoch * self.epochs) * 100
                self.progress_callback(self.job_id, progress)
            
            print(f'Training epoch {epoch + 1} done!')
            








