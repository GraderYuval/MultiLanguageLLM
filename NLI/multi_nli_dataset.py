import torch
from torch.utils.data import Dataset

class MNLIDataset(Dataset):
    def __init__(self, file_path):
        self.samples = self.load_data(file_path)  # Implement the load_data() method
        
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, index):
        return self.samples[index]
    
    def load_data(self, file_path):
        samples = []
        
        with open(file_path, 'r', encoding='utf-8') as file:
            next(file)  # Skip header line
            for line in file:
                line = line.strip().split('\t')
                # sentence1 = line[8]  # Index of sentence1 column
                # sentence2 = line[9]  # Index of sentence2 column
                # label = line[11]     # Index of label column
                sentence1 = line[5]  # Index of sentence1 column
                sentence2 = line[6]  # Index of sentence2 column
                label = line[10]     # Index of label column
                
                samples.append((sentence1, sentence2, label))
                
        return samples
