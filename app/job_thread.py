import threading
import time

class JobThread(threading.Thread):
    def __init__(self):
        super().__init__()
        self.progress = 0
    
    def run(self):
        # doing stuff
        for _ in range(10):
            time.sleep(1)
            self.progress += 10
