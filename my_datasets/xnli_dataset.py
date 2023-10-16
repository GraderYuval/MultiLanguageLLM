from my_datasets.helpers import *
from torch.utils.data import Dataset

class XNLIDataset(Dataset):
    def __init__(self, tokenizer, dataset, special_token='', target_language='english', translate=False, language=None):
        self.tokenizer = tokenizer
        self.premises = dataset["premise"]
        self.hypotheses = dataset["hypothesis"]
        self.labels = dataset["label"]
        self.special_token = special_token
        self.input_max_length = 200
        self.target_max_length = 4
        self.target_language = target_language
        self.translate = translate
        self.language = language
        
        
    def __len__(self):
        return len(self.premises)

    def __getitem__(self, index):
        premise = self.premises[index]
        hypothesis = self.hypotheses[index]
        label = self.labels[index]
        target = build_xnli_target(label, self.language)
        
        return {
            "input_ids":
                self.tokenizer(self._create_input_text(premise, hypothesis),
                               max_length=self.input_max_length,
                               padding="max_length",
                               truncation=True,
                               return_tensors="pt")["input_ids"].flatten(),
            "labels":
                self.tokenizer(target,
                               max_length=self.target_max_length,
                               padding="max_length",
                               truncation=True,
                               return_tensors="pt")["input_ids"].flatten(),
            "premise": premise,
            "hypothesis": hypothesis,
            "target": target
        }


    def _create_input_text(self, premise, hypothesis):
        instructions = []
        if self.translate:
            instructions.append(build_translation_prompt(self.target_language, self.language))
        instructions.extend(build_xnli_prompt(self.special_token, premise, hypothesis, self.language))
        return "\n".join(instructions)


class TwoStageXNLIDataset(XNLIDataset):
    def __init__(self, tokenizer, dataset, trans_model, special_token='', target_language='english', translate=False, language=None):
        self.tokenizer = tokenizer
        self.premises = dataset["premise"]
        self.hypotheses = dataset["hypothesis"]
        self.labels = dataset["label"]
        self.special_token = special_token
        self.input_max_length = 200
        self.target_max_length = 4
        self.target_language = target_language
        self.translate = translate
        self.language = language
        self.trans_model = trans_model
        
        
    def __len__(self):
        return len(self.premises)

    def __getitem__(self, index):
        

        premise = self.premises[index]
        hypothesis = self.hypotheses[index]
        label = self.labels[index]
        target = build_xnli_target(label, self.language)
        

        premise = self.trans_model(premise, max_length=512)[0]['translation_text'].strip('YY ')
        hypothesis = self.trans_model(hypothesis, max_length=512)[0]['translation_text'].strip('YY ')
        return {
            "input_ids":
                self.tokenizer(self._create_input_text(premise, hypothesis),
                               max_length=self.input_max_length,
                               padding="max_length",
                               truncation=True,
                               return_tensors="pt")["input_ids"].flatten(),
            "labels":
                self.tokenizer(target,
                               max_length=self.target_max_length,
                               padding="max_length",
                               truncation=True,
                               return_tensors="pt")["input_ids"].flatten(),
            "premise": premise,
            "hypothesis": hypothesis,
            "target": target
        }


    def _create_input_text(self, premise, hypothesis):
        instructions = []
        if self.translate:
            instructions.append(build_translation_prompt(self.target_language, self.language))
        instructions.extend(build_xnli_prompt(self.special_token, premise, hypothesis, self.language))
        return "\n".join(instructions)

