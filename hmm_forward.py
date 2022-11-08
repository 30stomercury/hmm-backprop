from torch.autograd import Function
import torch
import hmm_forward_cpp


class HMMForward(Function):
    @staticmethod
    def forward(ctx, potential, lengths, mask_pad):
        """
        potential : (B, T, N, N)
            The potential for computing messages.
        lengths : (B, N)
            The lengths in a batch.
        mask_pad : (B, T, 1, 1)
            The unpadded are masked with 0 while the paddings are 
            masked with log zeros, e.g. [0, 0, 0, -1e23, -1e23].
        """
        # mask the potential
        potential = potential * mask_pad.exp()
        chart, log_partition = hmm_forward_cpp.forward(potential)
        log_partition = torch.gather(log_partition, 1, lengths.view(-1, 1) - 1)
        ctx.save_for_backward(potential, chart, log_partition, mask_pad)
        return log_partition

    @staticmethod
    def backward(ctx, grad_z):
        potential, chart, log_partition, mask_pad = ctx.saved_tensors
        d_potential = hmm_forward_cpp.backward(
            grad_z, *ctx.saved_tensors)
        return -d_potential, None, None 
