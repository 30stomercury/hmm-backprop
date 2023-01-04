## A custom differentiable HMM layer 

Explicit implementation of the backward pass is technically not necessarry due to 
[Eisner's paper](https://aclanthology.org/W16-5901.pdf) (also see [Torch-Struct](https://github.com/harvardnlp/pytorch-struct)).
However, running dynamic programming in Pytorch is computationally expensive for long sequences such as speech.

This repo provides a C++ extension of the HMM layer, gradients are computed with **batched** message passing algorithm (also known as forward-backward).

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
        [[1, 1, 1, 0, 0], 
         [1, 1, 1, 1, 1]], device=device)

# prepare potential (can be replaced with neural nets)
emission = -1 * torch.randn(batch, time, num_states, num_states, 
    device=device, requires_grad=True).pow(2)
transition = torch.randn(batch, time, num_states, num_states, 
    device=device).log_softmax(-1)
potential = emission + transition

# forward
mask_pad = mask_pad.view(batch, time, 1, 1)
log_partition = HMMForward.apply(potential, lengths, mask_pad)
loss = - log_partition.sum()
```

### More examples
More examples will be added.


### GPU Performance
Benchmarked on a single NVIDIA 1080 Ti. The improvements are more pronounced when 
running on longer sequences.

**Forward pass:**
| *Batch*=8, *N*=50                 | hmm-backprop  | Pytorch |
|-----------------------------------|-------|---------|
| *L*=100                           | 30.6 ms| 39.3 ms   |
| *L*=500                           | 118.0 ms| 145.5 ms   |
| *L*=1000                          | 185.3 ms| 270.0 ms   |
| *L*=2000                          | 339.9 ms| 526.3 ms   |
| *L*=3000                          | 486.1 ms| 800.5 ms   |

**Backward pass:**
| *Batch*=8, *N*=50                 | hmm-backprop  | Pytorch |
|-----------------------------------|-------|---------|
| *L*=100                           | 37.4 ms| 55.2 ms   |
| *L*=500                           | 141.8 ms| 368.2 ms   |
| *L*=1000                          | 224.2 ms| 1528.4 ms   |
| *L*=2000                          | 397.1 ms| 6680.4 ms   |
| *L*=3000                          | 566.5 ms| 13962.0 ms   |

### Citation
```
@article{yeh2022learning,
  title={Learning Dependencies of Discrete Speech Representations with Neural Hidden Markov Models},
  author={Yeh, Sung-Lin and Tang, Hao},
  journal={arXiv preprint arXiv:2210.16659},
  year={2022}
}

```
