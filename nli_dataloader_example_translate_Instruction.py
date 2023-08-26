import argparse
import torch
import pandas as pd
from NLI.multi_nli_dataset import MNLIDataset


def create_argsparse():
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset_path',
                        type=str,
                        help='Path for MNLI data text file.')

    parser.add_argument('--batch_size',
                        type=int,
                        default=2,
                        help='Size of batch')

    parser.add_argument('--target_language',
                        type=str,
                        default="english",
                        help='Target language for translation')

    return parser.parse_args()


args = create_argsparse()

# dataset download from https://github.com/facebookresearch/XNLI
dataset = MNLIDataset(file_path=args.dataset_path)
data_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

# Initialize empty lists to hold input and target texts
input_texts = []
target_texts = []

for batch in data_loader:
    prompt_batch, (sentence1_batch, sentence2_batch, label_batch) = batch

    # Add translation instruction with additional details to the prompt
    translation_instruction = (f"Translate the following text into {args.target_language}. "
                               "Make sure your translation is accurate. "
                               "Make sure you are computing the task based on the translated language: ")
    prompt_batch = [translation_instruction + prompt for prompt in prompt_batch]

    results = label_batch  # should be replaced by LLM answer
    # TODO: calculate results accuracy (maybe after some postprocess) vs label_batch

    input_texts.extend(prompt_batch)
    target_texts.extend(label_batch)

# Create DataFrame
df = pd.DataFrame(data={'input_text': input_texts, 'target_text': target_texts})

# Save DataFrame as a tsv file
df.to_csv('NlI/data/xnli_input_output_data.tsv', sep='\t', index=False)
