import argparse
import torch

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('input_checkpoint', type=str)
parser.add_argument('output_model', type=str)
parser.add_argument('--state_dict_key', type=str, default='state_dict')

args = parser.parse_args()

checkpoint = torch.load(args.input_checkpoint)

state = dict()
for key, weight in checkpoint[args.state_dict_key].items():
	new_key = key.replace('module.','')
	print('{}->{}'.format(key,new_key))
	state[new_key] = weight

torch.save(state, args.output_model)