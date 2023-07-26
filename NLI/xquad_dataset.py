import torch
from torch.utils.data import Dataset

import json

class XQUADDataset(Dataset):
    def __init__(self, file_path):
        self.samples = self.load_data(file_path) 
        self.desc = \
            "Answer the question from the given passage. Your answer should be directly extracted from the passage, and it should be a single entity, name, or number, not a sentence."
        self.passage_template = "Passage: $$Passage$$"
        self.question_template = "Question: $$Question$$"
        self.note = "Note: Your answer should be directly extracted from the passage and be a single entity, name, or number, not as entence"
        
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, index):
        task = "\n".join([
            self.desc,
            self.passage_template.replace("$$Passage$$", self.samples[index][0]),
            self.question_template.replace("$$Question$$", self.samples[index][1]),
            self.note
        ])

        return task, self.samples[index]
    
    def load_data(self, file_path):
        samples = []
        
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)['data']
            for title in data:
                for paragraph in title['paragraphs']:
                    context = paragraph['context']
                    for q_a in paragraph['qas']:
                        question = q_a['question']
                        assert len(q_a["answers"]) == 1
                        answer = q_a["answers"][0]["text"]
                        samples.append((context, question, answer))
                
        return samples
