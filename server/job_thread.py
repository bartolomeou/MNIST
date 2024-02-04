import threading
import time

class JobThread(threading.Thread):
    def __init__(self, job_id, progress_callback):
        super().__init__()
        self.job_id = job_id
        self.progress_callback = progress_callback
        self.progress = 0
    
    def run(self):
        # doing stuff
        for _ in range(10):
            time.sleep(1)
            self.progress += 10
            self.progress_callback(self.job_id, self.progress)



