# A custom HMM layer that supports backpropagation

Explicit implementation of the backward pass is technically not required due to 
[Eisner's paper](https://aclanthology.org/W16-5901.pdf) (also see [Torch-Struct](https://github.com/harvardnlp/pytorch-struct)).
However, running dynamic programming in Pytorch is computationally expensive for long sequences such as speech.

This repo provides an efficient HMM layer that avoids Python for-loop.
