from torch.utils.data import Dataset

class XNLIDataset(Dataset):
    def __init__(self, tokenizer, dataset, special_token='', input_max_length=100, target_max_length=10, target_language='english',
                 translate=False):
        self.tokenizer = tokenizer
        self.premises = dataset["premise"]
        self.hypotheses = dataset["hypothesis"]
        self.labels = dataset["label"]
        self.special_token = special_token
        self.input_max_length = input_max_length
        self.target_max_length = target_max_length
        self.target_language = target_language
        self.translate = translate  # added translate flag

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
                self.tokenizer(self._create_target_text(label),
                               max_length=self.target_max_length,
                               padding="max_length",
                               truncation=True,
                               return_tensors="pt")["input_ids"].flatten(),
        }

    def _create_input_text(self, premise, hypothesis):
        instructions = []

        # Add translation instruction if translate is True
        if self.translate:
            instructions.append(f"Translate the following text into {self.target_language}. "
                                "Make sure your translation is accurate. "
                                "Then, based on the translated text, compute the task:")

        # Add main task instruction
        instructions.extend([
            f"{self.special_token} Please identify whether the premise entails or contradicts "
            "the hypothesis in the following premise and hypothesis. "
            "The answer should be exact \"entailment\", \"contradiction\", or \"neutral\".",
            f"Premise: {premise}",
            f"Hypothesis: {hypothesis}"
        ])

        return "\n".join(instructions)

@staticmethod
    def _create_target_text(label):
        match label:
            case 0:
                return "entailment"
            case 1:
                return "neutral"
            case 2:
                return "contradiction"
            case _:
                raise ValueError(f"{label=} must be 0, 1 or 2")
