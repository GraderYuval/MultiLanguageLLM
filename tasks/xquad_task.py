import torch
import random
import datasets
from tasks.base_task import BaseTask
from torch.utils.data import DataLoader
from my_datasets.xquad_dataset import XQUADDataset

class XQUADTask(BaseTask):
    @staticmethod
    def _get_task_name():
        return "xquad"
    
    def _init_data_loaders(self):
        dataset = datasets.load_dataset(self._get_task_name(), f"{self._get_task_name()}.{self.args.data_language}")[datasets.Split.VALIDATION]
        
        n = len(dataset)
        s1 = round(0.95 * n)
        s2 = round(0.02 * n)
        
        indices = random.sample(range(n), n)
        train_indices = indices[:s1] # 95%
        val_indices = indices[s1:s1 + s2] # 2%
        test_indices = indices[s1 + s2:] # 3%
        
        train_dataset = [dataset[i] for i in train_indices]
        val_dataset = [dataset[i] for i in val_indices]
        test_dataset = [dataset[i] for i in test_indices]
        
        train_dataset = self._update_dataset(train_dataset)
        val_dataset = self._update_dataset(val_dataset)
        test_dataset = self._update_dataset(test_dataset)
        
        train_dataset = XQUADDataset(self.tokenizer, train_dataset, special_token=self.args.special_token, language=self.args.data_language)
        self.train_data_loader = DataLoader(train_dataset, batch_size=self.args.batch_size, shuffle=True)
        val_dataset = XQUADDataset(self.tokenizer, val_dataset, special_token=self.args.special_token, language=self.args.data_language)
        self.val_data_loader = DataLoader(val_dataset, shuffle=True)
        test_dataset = XQUADDataset(self.tokenizer, test_dataset, special_token=self.args.special_token, language=self.args.data_language)
        self.test_data_loader = DataLoader(test_dataset, shuffle=True)
        
    @staticmethod
    def _update_dataset(input_dataset):
        output_dataset = {"contexts": [], "questions": [], "answers": []}
        for row in input_dataset:
            for answer in row["answers"]["text"]:
                output_dataset["contexts"].append(row["context"])
                output_dataset["questions"].append(row["question"])
                output_dataset["answers"].append(answer)
        return output_dataset
        
    def _test_example(self, example, idx):
        input_ids = example["input_ids"].to(self.device)
        context = example["context"]
        question = example["question"]
        answer = example["answer"]

        with torch.no_grad():
            output = self.model.generate(input_ids)
        output = self.tokenizer.decode(output[0], skip_special_tokens=True)
        
        print(f"Test {idx}: Context: {context}")
        print(f"Question: {question}")
        print(f"Answer: {answer}")
        print(f"Generated Output: {output}")
        print("=" * 50)
        
        return output == answer
    
if __name__ == "__main__":
    task = XQUADTask()
    task.run()