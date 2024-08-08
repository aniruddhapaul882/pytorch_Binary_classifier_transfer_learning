import os

class Config():
    
    RESIZE_SIZE = int(os.getenv('RESIZE_SIZE', 256))
    BATCH_SIZE = int(os.getenv('BATCH_SIZE', 32))
    DATA_DIR = str(os.getenv('DATA_DIR',"Harness_dataset_split"))
    NUM_WORKERS = int(os.getenv('NUM_WORKERS', os.cpu_count()))
    NUM_EPOCHS = int(os.getenv('NUM_EPOCHS', 2))
    MODEL_PATH = str(os.getenv('MODEL_PATH',"models/"))
    PRED_THRESH = float(os.getenv('PRED_THRESH', 0.5))
    WEIGHT_DIR = str(os.getenv('WEIGHT_DIR',"./weight"))
    RESULT_DIR = str(os.getenv('RESULT_DIR',"./result"))
    LEARNING_RATE = float(os.getenv('LEARNING_RATE', 0.001))
    STEP_SIZE = int(os.getenv('STEP_SIZE', 7))
    GAMMA = float(os.getenv('GAMMA', 0.1))
