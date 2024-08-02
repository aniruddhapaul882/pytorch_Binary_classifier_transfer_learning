import os

class Config():
    RESIZE_SIZE = int(os.getenv('RESIZE_SIZE', 256))
    BATCH_SIZE = int(os.getenv('BATCH_SIZE', 32))
    DATA_DIR = str(os.getenv('DATA_DIR',"data/hymenoptera_data"))
    NUM_WORKERS = int(os.getenv('NUM_WORKERS', os.cpu_count()))