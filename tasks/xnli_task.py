import torch
import datasets
from tasks.base_task import BaseTask
from torch.utils.data import DataLoader
from my_datasets.xnli_dataset import XNLIDataset, TwoStageXNLIDataset

class XNLITask(BaseTask):
    @staticmethod
    def _get_task_name():
        return "xnli"
    
    def _init_data_loaders(self):
        dataset = datasets.load_dataset(self._get_task_name(), self.args.data_language)
        train_dataset = TwoStageXNLIDataset(self.tokenizer, dataset[datasets.Split.TRAIN],
                                    special_token=self.args.special_token,
                                    target_language='english',
                                    translate=False,
                                    language=self.args.data_language,
                                    trans_model=self.trans_model

                                    )
        self.train_data_loader = DataLoader(train_dataset, batch_size=self.args.batch_size, shuffle=True)
        val_dataset = TwoStageXNLIDataset(self.tokenizer, dataset[datasets.Split.VALIDATION],
                                  special_token=self.args.special_token,
                                  target_language='english',
                                  translate=False,
                                  language=self.args.data_language,
                                  trans_model=self.trans_model
                                  )
        self.val_data_loader = DataLoader(val_dataset, shuffle=True)
        test_dataset = TwoStageXNLIDataset(self.tokenizer, dataset[datasets.Split.TEST],
                                    special_token=self.args.special_token,
                                   target_language='english',
                                   translate=False,
                                   language=self.args.data_language,
                                    trans_model=self.trans_model
                                   )
        self.test_data_loader = DataLoader(test_dataset, shuffle=True)
        
    def _test_example(self, example, idx):
        input_ids = example["input_ids"].to(self.device)
        premise = example["premise"]
        hypothesis = example["hypothesis"]
        target = example["target"]

        with torch.no_grad():
            output = self.model.generate(input_ids)
        output = self.tokenizer.decode(output[0], skip_special_tokens=True)
        
        print(f"Test {idx}: Premise: {premise}")
        print(f"Target: {target}")
        print(f"Hypothesis: {hypothesis}")
        print(f"Generated Output: {output}")
        print("=" * 50)
        
        return output == target
    
if __name__ == "__main__":
    task = XNLITask()
    task.run()