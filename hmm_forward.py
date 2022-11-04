from torch.autograd import Function
import torch
import hmm_forward_cpp

torch.manual_seed(42)


class HMMForward(Function):
    @staticmethod
    def forward(ctx, potential, length, mask):
        chart, partition = hmm_forward_cpp.forward(potential)
        partition = torch.gather(partition, 1, length.view(-1, 1) - 1)
        ctx.save_for_backward(potential, chart, partition, mask)
        return partition

    @staticmethod
    def backward(ctx, grad_z):
        potential, chart, partition, mask = ctx.saved_variables
        d_potential = hmm_forward_cpp.backward(
            grad_z, *ctx.saved_variables)
        print(d_potential[0].sum(-1))
        return -d_potential, None, None 

