import argparse
import torch
from NLI.xquad_dataset import XQUADDataset


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

# dataset download https://github.com/deepmind/xquad
dataset = XQUADDataset(file_path=args.dataset_path)
data_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

for batch in data_loader:
    prompt_batch, (context_batch, question_batch, answer_batch) = batch

    # Add translation instruction with additional details to the prompt
    translation_instruction = (f"Translate the following text into {args.target_language}. "
                               "Make sure your translation is accurate. "
                               "Then, based on the translated text, compute the task: ")
    prompt_batch = [translation_instruction + prompt for prompt in prompt_batch]

    results = answer_batch  # should be replaced by LLM answer

    # TODO: calculate results accuracy (maybe after some postprocess) vs answer_batch
