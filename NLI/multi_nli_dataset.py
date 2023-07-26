import torch
from torch.utils.data import Dataset

class MNLIDataset(Dataset):
    def __init__(self, file_path):
        self.samples = self.load_data(file_path) 
        self.desc = \
            "Please identify whether the premise entails or contradicts the hypothesis in the following premise and hypothesis. The answer should be exact \"entailment\", \"contradiction\", or \"neutral\"."
        self.premise_template = "Premise: $$Premise$$"
        self.hypothesis_template = "Hypothesis: $$Hypothesis$$"

        
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, index):
        task = "\n".join([
            self.desc,
            self.premise_template.replace("$$Premise$$", self.samples[index][0]),
            self.hypothesis_template.replace("$$Hypothesis$$", self.samples[index][1])
        ])

        return task, self.samples[index]
    
    def load_data(self, file_path):
        samples = []
        
        with open(file_path, 'r', encoding='utf-8') as file:
            next(file)  # Skip header line
            for line in file:
                line = line.strip().split('\t')

                sentence1, sentence2, label = line
                
                samples.append((sentence1, sentence2, label))
                
        return samples
