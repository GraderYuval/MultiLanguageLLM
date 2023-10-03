from my_datasets.helpers import *
from torch.utils.data import Dataset

class XQUADDataset(Dataset):
    def __init__(self, tokenizer, dataset, special_token='', target_language='english', translate=False, language=None):
        self.tokenizer = tokenizer
        self.contexts = dataset["contexts"]
        self.questions = dataset["questions"]
        self.answers = dataset["answers"]
        self.special_token = special_token
        self.input_max_length = 1000
        self.target_max_length = 100
        self.target_language = target_language
        self.translate = translate
        self.language = language
        
        
    def __len__(self):
        return len(self.contexts)

    def __getitem__(self, index):
        context = self.contexts[index]
        question = self.questions[index]
        answer = self.answers[index]

        return {
            "input_ids":
                self.tokenizer(self._create_input_text(context, question),
                               max_length=self.input_max_length,
                               padding="max_length",
                               truncation=True,
                               return_tensors="pt")["input_ids"].flatten(),
            "labels":
                self.tokenizer(answer,
                               max_length=self.target_max_length,
                               padding="max_length",
                               truncation=True,
                               return_tensors="pt")["input_ids"].flatten(),
            "context": context,
            "question": question,
            "answer": answer
        }


    def _create_input_text(self, context, question):
        instructions = []
        if self.translate:
            instructions.append(build_translation_prompt(self.target_language, self.language))
        instructions.extend(build_xquad_prompt(self.special_token, context, question, self.language))
        return "\n".join(instructions)
