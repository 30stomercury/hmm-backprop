## A custom differentiable HMM layer for end-to-end neural HMMs

Explicit implementation of the backward pass is technically not required due to 
[Eisner's paper](https://aclanthology.org/W16-5901.pdf) (also see [Torch-Struct](https://github.com/harvardnlp/pytorch-struct)).
However, running dynamic programming in Pytorch is computationally expensive for long sequences such as speech.

This repo provides a C++ extension of the HMM layer that avoids Python for-loop.

### How to install
To build C++ extension, run:
```
python setup.py install
```

### How to use
The Pytorch interface is defined in `hmm_forward.py`.
```python
import torch
from hmm_forward import HMMForward

device = torch.device("cuda")
batch = 2
time = 5
num_states = 10
lengths = torch.tensor([3, 5], device=device, dtype=torch.int64)
mask_pad = torch.tensor(
        [[0, 0, 0, 1, 1], 
         [0, 0, 0, 0, 0]], device=device) * -1e23

# prepare potential (can be replaced with neural nets)
emission = -1 * torch.randn(batch, time, num_states, num_states, 
    device=device, requires_grad=True).pow(2)
transition = torch.randn(batch, time, num_states, num_states, 
    device=device).log_softmax(-1)
potential = emission + transition

# forward
mask_pad = mask_pad.view(batch, time, 1, 1)
partition = HMMForward.apply(potential, lengths, mask_pad)
loss = - partition.sum()
```

### Citation
```
@article{yeh2022learning,
  title={Learning Dependencies of Discrete Speech Representations with Neural Hidden Markov Models},
  author={Yeh, Sung-Lin and Tang, Hao},
  journal={arXiv preprint arXiv:2210.16659},
  year={2022}
}

```
