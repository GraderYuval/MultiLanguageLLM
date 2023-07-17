import torch
from NLI.multi_nli_dataset import MNLIDataset

# dataset download from https://cims.nyu.edu/~sbowman/multinli/multinli_1.0.zip

dataset = MNLIDataset(file_path=r"multinli_1.0\multinli_1.0\multinli_1.0_train.txt")
data_loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

for batch in data_loader:
    sentence1_batch, sentence2_batch, label_batch = batch
    aa=2

