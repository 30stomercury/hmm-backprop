#include <torch/extension.h>
#include <vector>
using namespace torch::indexing;

double logzero = -1e23;


/* Sum operation in Log Semiring
 * Matrix multiplication in log space.

Arguments
---------
log_a: (B, N) 
log_b: (B, N, N)

Returns
-------
(B, N)
*/
torch::Tensor log_matmul(
    torch::Tensor a, const torch::Tensor& b) {
    auto unsqueezed = a.unsqueeze(-2);
    return torch::logsumexp(unsqueezed + b, /*dim=*/{-1}, false);
}


/* Forward algorithm in log space (forward).

Arguments
---------
potential: (B, T, N, N) 

Returns
-------
{(B, T, N), (B, N)}
*/
std::vector<torch::Tensor> hmm_fw_forward(const torch::Tensor& potential) {
    const int batch = potential.sizes()[0];
    const int time = potential.sizes()[1];
    const int N = potential.sizes()[2];
    auto chart = torch::zeros({batch, time, N}, potential.device());

    chart.index_put_({Slice(), 0}, potential.index({Slice(), 0, 0}));
    auto potential_ = potential.transpose(-2, -1);
    for (int64_t i = 1; i < time; ++i) {
        chart.index({Slice(), i}) = 
            log_matmul(
                chart.index({Slice(), i-1}), 
                potential_.index({Slice(), i})
            );
    }
    // (B, T)
    auto log_partition = torch::logsumexp(chart, /*dim=*/{-1});

    return {chart, log_partition};
}


/* Backward algorithm in log space.

Arguments
---------
potential: (B, T, N, N) 

Returns
-------
(B, N)
*/
torch::Tensor hmm_bw_forward(const torch::Tensor& potential) {
    const int batch = potential.sizes()[0];
    const int time = potential.sizes()[1];
    const int N = potential.sizes()[2];
    // Init chart with zeros
    auto chart = torch::zeros({batch, time, N}, potential.device());

    int64_t start = time - 2;
    for (int64_t i = start; i > -1; --i) {
        chart.index({Slice(), i}) = 
            log_matmul(chart.index({Slice(), i+1}), 
                       potential.index({Slice(), i+1})
        ).clamp_(logzero, 0);
    }

    return chart;
}

/* Forward-Backward algorithm in log space (Backward).
*/
torch::Tensor hmm_fw_backward(
        const torch::Tensor& grad_z, 
        const torch::Tensor& potential, 
        const torch::Tensor& fwd_vars,
        const torch::Tensor& log_partition,
        const torch::Tensor& mask) {
    const int batch = potential.sizes()[0];
    const int time = potential.sizes()[1];
    const int N = potential.sizes()[2];
    torch::Tensor chart = torch::zeros({batch, time, N, N}, potential.device());
    // fwd and bwd vars
    torch::Tensor bwd_vars = hmm_bw_forward(potential);
    torch::Tensor bwd_vars_ = bwd_vars.unsqueeze(-2);
    torch::Tensor fwd_vars_ = fwd_vars.unsqueeze(-1);
    
    int64_t end = time - 1;
    chart.index_put_({Slice(), 0}, logzero);
    chart.index_put_({Slice(), 0, 0}, 
        fwd_vars.index({Slice(), 0}) + 
        bwd_vars.index({Slice(), 0})
    );
    for (int64_t i = 0; i < end; ++i) {
        chart.index_put_(
            {Slice(), i+1},
            fwd_vars_.index({Slice(), i}) + 
            potential.index({Slice(), i+1}) +
            bwd_vars_.index({Slice(), i+1})
        );
    }
    torch::Tensor grad = 
        chart.clamp_(logzero, 0)
        - log_partition.view({batch, 1, 1, 1})
        + (-grad_z).log().view({batch, 1, 1, 1});
    grad.masked_fill_(mask==0, logzero);

    return grad.exp();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &hmm_fw_forward, "HMM forward");
  m.def("backward", &hmm_fw_backward, "HMM backward");
}
