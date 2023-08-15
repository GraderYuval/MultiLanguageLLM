import torch
import argparse

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task",
                        type=str,
                        default="mnli")
    parser.add_argument("--model_name",
                        type=str,
                        default="google/mt5-small")
    parser.add_argument("--special_token",
                        type=str,
                        default="<idf.lang>")
    parser.add_argument("--batch_size",
                        type=int,
                        default=8)
    parser.add_argument("--input_max_length",
                        type=int,
                        default=200)
    parser.add_argument("--target_max_length",
                        type=int,
                        default=4)
    parser.add_argument("--lr",
                        type=float,
                        default=1e-5)
    parser.add_argument("--epochs",
                        type=int,
                        default=1)
    parser.add_argument("--checkpoint",
                        type=int,
                        default=10)
    return parser.parse_args()


def checkpoint(data_loader, device, model, tokenizer, train_avg_loss):
    for batch_idx, batch in enumerate(data_loader):
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        outputs = model(input_ids=input_ids, labels=labels)
        loss = outputs.loss
        input_id = input_ids[0]
        label = model.generate(input_id.unsqueeze(0))[0]
        example_input_text = tokenizer.decode(input_id, skip_special_tokens=True)
        examp
             f"Validation Loss: {loss:.4f}")le_target_text = tokenizer.decode(label, skip_special_tokens=True)
        print(f"Epoch {epoch + 1}/{args.epochs}, "
             f"Step {batch_idx + 1}/{len(train_data_loader)}, "
             f"Train Loss: {train_avg_loss:.4f}, "
        print(f"Example Input: {example_input_text}")
        print(f"Example Output: {example_target_text}")
        torch.save(model.state_dict(), model_path.joinpath(f"epoch_{epoch + 1}_step_{batch_idx + 1}.pt"))
        return loss
