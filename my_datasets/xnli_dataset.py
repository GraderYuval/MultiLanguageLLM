from my_datasets.helpers import *
from torch.utils.data import Dataset

class XNLIDataset(Dataset):
    def __init__(self, tokenizer, dataset, special_token='', input_max_length=100, target_max_length=10, target_language='english',
                 translate=False, language=None):
        self.tokenizer = tokenizer
        self.premises = dataset["premise"]
        self.hypotheses = dataset["hypothesis"]
        self.labels = dataset["label"]
        self.special_token = special_token
        self.input_max_length = input_max_length
        self.target_max_length = target_max_length
        self.target_language = target_language
        self.translate = translate
        self.language = language
        
        
    def __len__(self):
        return len(self.premises)

    def __getitem__(self, index):
        premise = self.premises[index]
        hypothesis = self.hypotheses[index]
        label = self.labels[index]

        return {
            "input_ids":
                self.tokenizer(self._create_input_text(premise, hypothesis),
                               max_length=self.input_max_length,
                               padding="max_length",
                               truncation=True,
                               return_tensors="pt")["input_ids"].flatten(),
            "labels":
                self.tokenizer(build_xnli_target(label, self.language),
                               max_length=self.target_max_length,
                               padding="max_length",
                               truncation=True,
                               return_tensors="pt")["input_ids"].flatten(),
            "premise": premise,
            "hypothesis": hypothesis
        }


    def _create_input_text(self, premise, hypothesis):
        instructions = []
        if self.translate:
            instructions.append(build_translation_prompt(self.target_language, self.language))
        instructions.extend(build_xnli_prompt(premise, hypothesis, self.language))
        return "\n".join(instructions)
