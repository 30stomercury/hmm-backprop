from torch.autograd import Function
import torch
import hmm_forward_cpp


class HMMForward(Function):
    @staticmethod
    def forward(ctx, potential, lengths, mask):
        """
        potential : (B, T, N, N)
            The potential for computing messages.
        lengths : (B, N)
            The lengths in a batch.
        mask: (B, T, 1, 1)
            The padded are masked with 0 while the paddings are 
            masked with 1, e.g. [1, 1, 1, 0, 0].
        """
        # for the backward algorithm
        potential.masked_fill_(mask==0, 1e23)
        chart, log_partition = hmm_forward_cpp.forward(potential)
        log_partition = torch.gather(log_partition, 1, lengths.view(-1, 1) - 1)
        ctx.save_for_backward(potential, chart, log_partition, mask)
        return log_partition

    @staticmethod
    def backward(ctx, grad_z):
        potential, chart, log_partition, mask_pad = ctx.saved_tensors
        d_potential = hmm_forward_cpp.backward(
            grad_z, *ctx.saved_tensors)
        return -d_potential, None, None 
