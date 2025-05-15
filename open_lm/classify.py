import torch
from open_lm.params import parse_args
import argparse
from open_lm.model import test_classif_model
from collections import Counter

# Parse arguments
args = parse_args([])
parser = argparse.ArgumentParser(description="Override params arguments with command-line arguments")
parser.add_argument('--model', type=str, help='Model name to use for evaluation')
parser.add_argument('--classif-model-path', type=str, help='Path to the classification model checkpoint')
parser.add_argument('--num-classes', type=int, required=True, help='Number of classes for evaluation')
parser.add_argument('--generated-data-path', type=str, required=True, help='Path for the generated data. A list of pytorch tensors is expected.')
cmd_args = parser.parse_args()

args.model = cmd_args.model
args.classif_model_path = cmd_args.classif_model_path
args.num_classes = cmd_args.num_classes

# Load model and dataset
model = test_classif_model(args)
model = model.to('cuda')
dataset = torch.load(base_path)

pred = []
for sample in dataset:
    sample = torch.LongTensor(sample).to(device)
    with torch.no_grad():
        out, _, _ = model(sample) 
        pred.append(torch.argmax(out,2)[:,-1].item())

# Count occurrences of each class using Counter
class_counts = Counter(pred)

l = len(dataset)

# Calculate and print percentages for each class
for i in range(args.num_classes):
    percentage = class_counts[i] * 100 / l
    print(f"class{i}: {percentage:.2f}%")


