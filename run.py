
import os
from train import main as train_main

if __name__ == '__main__':
    # Ensure the necessary directories exist
    os.makedirs(cfg.WEIGHT_DIR, exist_ok=True)
    os.makedirs(cfg.RESULT_DIR, exist_ok=True)
    
    # Execute the training process
    train_main()
