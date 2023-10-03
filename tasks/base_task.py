import os
import glob
import tqdm
import torch
import argparse
import datasets
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from my_datasets.xnli_dataset import XNLIDataset
from transformers import MT5Tokenizer, MT5ForConditionalGeneration



class BaseTask:
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.args = None
        self.model_path = None
        self.tokenizer = None
        self.device = None
        self.model = None
        self.train_data_loader = None
        self.val_data_loader = None
        self.test_data_loader = None
        self.train_losses = None
        self.val_losses = None
        self.val_steps = None
        self.optimizer = None
        self.loss_fn = None
        self.last_epoch = 0

    def run(self):
        self._init()
        if self.args.only_test == 0:
            self._train()
            self._plot()
        
        self._test()

    def _parse_args(self):
        self.parser.add_argument("--task",
                                 type=str,
                                 default="xnli")
        self.parser.add_argument("--data_language",
                                 type=str,
                                 default="en")
        self.parser.add_argument("--language",
                                 type=str,
                                 default="en")
        self.parser.add_argument("--model_name",
                                 type=str,
                                 default="google/mt5-small")
        self.parser.add_argument("--special_token",
                                 type=str,
                                 default="<idf.lang>")
        self.parser.add_argument("--batch_size",
                                 type=int,
                                 default=8)
        self.parser.add_argument("--input_max_length",
                                 type=int,
                                 default=200)
        self.parser.add_argument("--target_max_length",
                                 type=int,
                                 default=4)
        self.parser.add_argument("--lr",
                                 type=float,
                                 default=1e-5)
        self.parser.add_argument("--epochs",
                                 type=int,
                                 default=1)
        self.parser.add_argument("--val_steps_scale",
                                 type=int,
                                 default=10000)
        self.parser.add_argument("--ckpt_dir",
                                 type=str,
                                 default="./models")
        self.parser.add_argument("--only_test",
                                 type=int,
                                 default=0
                                 )

        return self.parser.parse_args()

    def _init(self):
        self.args = self._parse_args()
        print(f'{self.args=}')

        self.tokenizer = MT5Tokenizer.from_pretrained(self.args.model_name, legacy=False, cache_dir = "/home/dcor/nadavorzech/source/MultiLanguageLLM/.cahce")
        self.tokenizer.add_special_tokens(special_tokens_dict={"additional_special_tokens": [self.args.special_token]})
        model = MT5ForConditionalGeneration.from_pretrained(self.args.model_name, cache_dir = "/home/dcor/nadavorzech/source/MultiLanguageLLM/.cahce")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f'{self.device=}')
        self.model = model.to(self.device)

        self.model_path = Path(f"{self.args.ckpt_dir}/{self.args.task}/{self.args.language}/{self.args.model_name}")
        self.model_path.mkdir(parents=True, exist_ok=True)
        self._load_model_weights()

        dataset = datasets.load_dataset(self.args.task, self.args.data_language, cache_dir = "/home/dcor/nadavorzech/source/MultiLanguageLLM/.cahce")
        train_dataset = XNLIDataset(self.tokenizer, dataset[datasets.Split.TRAIN],
                                    special_token=self.args.special_token,
                                    input_max_length=self.args.input_max_length,
                                    target_max_length=self.args.target_max_length,
                                    target_language='english',
                                    translate=False,
                                    language=self.args.data_language
                                    )
        self.train_data_loader = DataLoader(train_dataset, batch_size=self.args.batch_size, shuffle=True)
        val_dataset = XNLIDataset(self.tokenizer, dataset[datasets.Split.VALIDATION],
                                  special_token=self.args.special_token,
                                  input_max_length=self.args.input_max_length,
                                  target_max_length=self.args.target_max_length,
                                  target_language='english',
                                  translate=False,
                                  language=self.args.data_language
                                  )
        self.val_data_loader = DataLoader(val_dataset, shuffle=True)
        test_dataset = XNLIDataset(self.tokenizer, dataset[datasets.Split.TEST],
                                    special_token=self.args.special_token,
                                    input_max_length=self.args.input_max_length,
                                    target_max_length=self.args.target_max_length,
                                   target_language='english',
                                   translate=False,
                                   language=self.args.data_language
                                   )
        self.test_data_loader = DataLoader(test_dataset, shuffle=True)

        self.train_losses = list()
        self.val_losses = list()
        self.val_steps = list()

        self.optimizer = torch.optim.AdamW(model.parameters(), lr=self.args.lr)
        self.loss_fn = torch.nn.CrossEntropyLoss()

    def _train(self):
        val_data_loader_it = iter(self.val_data_loader)
        for epoch in range(self.last_epoch, self.args.epochs):
            for batch_idx, batch in tqdm.tqdm(enumerate(self.train_data_loader)):
                input_ids = batch["input_ids"].to(self.device)
                labels = batch["labels"].to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(input_ids=input_ids, labels=labels)
                train_loss = outputs.loss
                self.train_losses.append(train_loss.item())

                train_loss.backward()
                self.optimizer.step()

                if batch_idx % self.args.val_steps_scale == 0 and batch_idx > 0:
                    val_batch = next(val_data_loader_it)
                    input_ids = val_batch["input_ids"].to(self.device)
                    labels = val_batch["labels"].to(self.device)

                    with torch.no_grad():
                        outputs = self.model(input_ids=input_ids, labels=labels)
                    val_loss = outputs.loss.detach().cpu().numpy()
                    self.val_losses.append(val_loss)
                    self.val_steps.append(batch_idx)

                    input_id = input_ids[0]
                    label = self.model.generate(input_id.unsqueeze(0))[0]
                    example_input_text = self.tokenizer.decode(input_id, skip_special_tokens=True)
                    example_target_text = self.tokenizer.decode(label, skip_special_tokens=True)

                    print(f"Epoch {epoch + 1}/{self.args.epochs}, Step {batch_idx}/{len(self.train_data_loader)}, "
                          f"Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}")
                    print(f"Example Input: {example_input_text}")
                    print(f"Example Output: {example_target_text}")

                    torch.save(self.model.state_dict(),
                               self.model_path.joinpath(f"epoch_{epoch + 1}_step_{batch_idx + 1}.pt"))

            print(f"Epoch {epoch + 1}/{self.args.epochs}, Loss: {np.mean(self.train_losses):.4f}")

    def _test(self):
        correct_predictions = 0
        for idx, example in tqdm.tqdm(enumerate(self.test_data_loader)):
            input_ids = example["input_ids"].to(self.device)
            premise = example["premise"]
            hypothesis = example["hypothesis"]
            labels = example["labels"].to(self.device)


            with torch.no_grad():
                output = self.model.generate(input_ids)
            output = self.tokenizer.decode(output[0], skip_special_tokens=True)

            if output == self.tokenizer.decode(labels[0], skip_special_tokens=True):
                correct_predictions += 1
            
            print(f"Test {idx}: Premise: {premise}")
            print(f"Hypothesis: {hypothesis}")
            print(f"Generated Output: {output}")
            print(f"Label: {labels}")
            print(f"Decoded Label: {self.tokenizer.decode(labels[0], skip_special_tokens=True)}")
            print("=" * 50)

        accuracy = correct_predictions / len(self.test_data_loader)
        print(f"Final Accuracy: {accuracy:.2%}")
    
    def _load_model_weights(self):
        
        ckpt_files_list = sorted(glob.glob(os.path.join(self.model_path, "*.pt")))
        if len(ckpt_files_list) == 0:
            return
        
        def sorting_key(item):
            return int(item[1]), int(item[3][:-3])
        
        ckpt_files_list = sorted([ckpt.split('_') for ckpt in ckpt_files_list], key=sorting_key)
        
        latest_ckpt_path = "_".join(ckpt_files_list[-1])
        print(latest_ckpt_path)
        ckpt_data = torch.load(latest_ckpt_path)
        self.model.load_state_dict(ckpt_data)
        self.last_epoch = int(latest_ckpt_path.split('_')[1])

    def _plot(self):
        plt.plot(self.train_losses, label='Training Loss')
        plt.plot(self.val_steps, self.val_losses, 'o', label='Validation Loss')
        plt.legend()
        plt.xlabel('Steps')
        plt.ylabel('Loss')
        plt.savefig(self.model_path.joinpath("Training and Validation Losses.png"))


if __name__ == "__main__":
    task = BaseTask()
    task.run()