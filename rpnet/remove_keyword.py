import torch

state_dict = torch.load('fh02_new_baseline.pth')
# create new OrderedDict that does not contain `module.`
from collections import OrderedDict
new_state_dict = OrderedDict()
for k, v in state_dict.items():
    if k.startswith('wR2'):
        new_key = 'wR2.' + k[11:]
        print('wR2.' + k[11:])
    else:
        new_key = k
        print(k)
    new_state_dict[new_key] = v
    #name = k[7:] # remove `module.`
    #new_state_dict[name] = v
# load params
#model.load_state_dict(new_state_dict)
torch.save(new_state_dict, 'fh02_newnew.pth')
