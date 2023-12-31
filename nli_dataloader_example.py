import argparse

import torch
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


    return parser.parse_args()



args = create_argsparse()

# dataset download from https://github.com/facebookresearch/XNLI
dataset = MNLIDataset(file_path=args.dataset_path)
data_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

for batch in data_loader:
    '''
        prompt - text descriptor to inject into LLM, corrently descriptor in english only
        sentence1 - Premise (probably needed only for debug)
        sentence2 - Hypothesis (probably needed only for debug)
        label - Labels from ['entailment', 'contradiction', 'neutral']
    '''
    
    prompt_batch, (sentence1_batch, sentence2_batch, label_batch) = batch
    # TODO: (NADAV) I tried to create new lines between each sentence. Check correctness when feeding the prompt to the LLM 

    results = label_batch # should be replace by LLM answer

    # calculate results accuracy (maybe after some postprocess) vs label_batch





