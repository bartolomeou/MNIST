import threading
import torch
from torch.utils.data import DataLoader

from .mnist import model, loss_fn, train_data, test_data


class JobThread(threading.Thread):
    def __init__(self, job_id, batch_size, learning_rate, epochs, progress_callback, completion_callback):
        super().__init__()
        self.job_id = job_id

        self.train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
        self.test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=True)
        self.optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
        self.epochs = epochs

        self.progress_callback = progress_callback
        self.completion_callback = completion_callback

        self.progress = 0


    def run(self):
        # train
        model.train()
        total_len = (len(self.train_dataloader.dataset) * self.epochs) + len(self.test_dataloader.dataset)

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
                self.progress += (len(X) / total_len) * 100
                self.progress_callback(self.job_id, self.progress)
            
            print(f'Training epoch {epoch + 1} done!')
        

        # test
        model.eval() 
        test_size = len(self.test_dataloader.dataset)
        num_batches = len(self.test_dataloader)
        test_loss, correct = 0, 0

        with torch.no_grad():
            for X, y in self.test_dataloader:
                pred = model(X)
                test_loss += loss_fn(pred,y).item()
                correct += (pred.argmax(1) == y).type(torch.float).sum().item()

                self.progress += (len(X) / total_len) * 100
                self.progress_callback(self.job_id, self.progress)
        
        test_loss /= num_batches
        accuracy = correct / test_size * 100

        # update database
        self.completion_callback(self.job_id, accuracy)

        self.progress = 100
        self.progress_callback(self.job_id, self.progress)
        