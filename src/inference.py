import numpy as np
import pandas as pd
import torch

from src import config, dataset, engine


def predict(df_test, model, device):
    test_dataset = dataset.TESTDataset(tweet=df_test[config.TEXT].values)
    test_data_loader = torch.utils.data.DataLoader(test_dataset, batch_size=config.TEST_BATCH_SIZE, num_workers=1)
    outputs = engine.train_fn(test_data_loader, model, device)
    outputs = np.array(outputs) >= config.THRESHOLD
    sample_submission = pd.read_csv(config.SUBMISSION)
    sample_submission["target"] = outputs
    sample_submission.to_csv(config.PREDICTION_PATH, index=False)
    return sample_submission
