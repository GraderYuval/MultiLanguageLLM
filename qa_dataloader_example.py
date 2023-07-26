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


    return parser.parse_args()



args = create_argsparse()

# dataset download https://github.com/deepmind/xquad
dataset = XQUADDataset(file_path=args.dataset_path)
data_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

for batch in data_loader:
    '''
        prompt - text descriptor to inject into LLM, corrently descriptor in english only
        sentence1 - Passage (probably needed only for debug)
        sentence2 - Question (probably needed only for debug)
        label - Answer for the question, single entity for example a name or a number
    '''
    
    prompt_batch, (context_batch, question_batch, answer_batch) = batch
    # TODO: (NADAV) I tried to create new lines between each sentence. Check correctness when feeding the prompt to the LLM 

    results = answer_batch # should be replace by LLM answer

    # calculate results accuracy (maybe after some postprocess) vs label_batch





