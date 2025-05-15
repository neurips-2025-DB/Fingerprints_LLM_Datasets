import torch
from open_lm.params import parse_args
import argparse
from open_lm.model import test_classif_model

# Parse arguments
args = parse_args([])
parser = argparse.ArgumentParser(description="Override params arguments with command-line arguments")
parser.add_argument('--model', type=str, help='Model name to use for evaluation')
parser.add_argument('--classif-model-path', type=str, help='Path to the classification model checkpoint')
parser.add_argument('--num-classes', type=int, required=True, help='Number of classes for evaluation')
parser.add_argument('--test-sets', type=str, nargs='+', required=True, help='Test set names (one for each class)')
parser.add_argument('--base-path', type=str, required=True, help='Base path for the test set files')
cmd_args = parser.parse_args()

args.model = cmd_args.model
args.classif_model_path = cmd_args.classif_model_path
args.num_classes = cmd_args.num_classes

# Ensure the number of test sets matches the number of classes
if len(cmd_args.test_sets) != args.num_classes:
    raise ValueError(f"Number of test sets ({len(cmd_args.test_sets)}) does not match num_classes ({args.num_classes}).")

# Load model
model = test_classif_model(args)
model = model.to('cuda')

# Evaluate over all classes
total_sum = 0
total_length = 0

for class_idx, test_set in enumerate(cmd_args.test_sets):
    data_path = cmd_args.base_path + '/' + test_set + '.pt'
    dataset = torch.load(data_path)
    
    class_sum = 0
    for sample in dataset:
        sample = torch.LongTensor(sample).to('cuda')
        
        with torch.no_grad():
            out, _, _ = model(sample)
            
            # Get predictions for the current class
            pred = torch.argmax(out, 2)[:, -1]
            
            # Count correct predictions for this class
            n_correct = torch.sum(pred == class_idx).item()
            class_sum += n_correct
    
    # Store results for this class
    total_sum += class_sum
    total_length += len(dataset)
    print(f"Class {class_idx} ({test_set}): {class_sum} / {len(dataset)}")

# Print overall results
print("Total= ", total_sum, "/", total_length)
print("Accuracy= ", total_sum / total_length * 100, "%")
