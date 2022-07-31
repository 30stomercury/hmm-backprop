#include <torch/extension.h>
#include <vector>
using namespace torch::indexing;


/* Sum operation in Max Semiring 

Arguments
---------
log_a: (B, N) 
log_b: (B, N, N)

Returns
-------
(B, N)
*/
torch::Tensor log_matmul(torch::Tensor a, const torch::Tensor& b) {
    auto unsqueezed = a.unsqueeze(-2);
    return torch::logsumexp(unsqueezed + b, /*dim=*/{-1}, false);
}

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
torch::Tensor log_matmul(torch::Tensor a, const torch::Tensor& b) {
    auto unsqueezed = a.unsqueeze(-2);
    return torch::logsumexp(unsqueezed + b, /*dim=*/{-1}, false);
}

/* Voterbi algorithm 

Arguments
---------
potential: (B, T, N, N) 

Returns
-------
(B, T, N)
*/
std::vector<torch::Tensor> hmm_viterbi(const torch::Tensor& potential) {
    const int batch = potential.sizes()[0];
    const int time = potential.sizes()[1];
    const int N = potential.sizes()[2];
    auto chart = torch::zeros({batch, time, N}, potential.device());
    double logzero = -1e23;
    chart.index_put_({Slice(), 0}, potential.index({Slice(), 0, 0}));
    auto potential_ = potential.transpose(-2, -1);
    for (int64_t i = 1; i < time; ++i) {
        chart.index({Slice(), i, Slice()}) = 
            log_matmul(
                chart.index({Slice(), i-1}), 
                potential_.index({Slice(), i})
            ).clamp(logzero, 0);
    }
    // (B, T)
    auto partition = torch::logsumexp(chart, /*dim=*/{-1});

    return {chart, partition};
}

/* Forward algorithm in log space.

Arguments
---------
potential: (B, T, N, N) 

Returns
-------
(B, N)
*/
std::vector<torch::Tensor> hmm_fw_forward(const torch::Tensor& potential) {
    const int batch = potential.sizes()[0];
    const int time = potential.sizes()[1];
    const int N = potential.sizes()[2];
    auto chart = torch::zeros({batch, time, N}, potential.device());
    double logzero = -1e23;
    /* pytorch syntax
    chart[:, 0] = potential[:, 0, :, 0]
    for t in range(1, T):
        chart[:, t] = self.log_matmul(chart[:, t-1], potential[:, t].transpose(-2, -1))
    log_partition = torch.logsumexp(chart, -1)
    log_partition = torch.gather(log_partition, 1, length.view(-1,1) - 1).squeeze(-1)
    */
    chart.index_put_({Slice(), 0}, potential.index({Slice(), 0, 0}));
    auto potential_ = potential.transpose(-2, -1);
    for (int64_t i = 1; i < time; ++i) {
        chart.index({Slice(), i, Slice()}) = 
            log_matmul(
                chart.index({Slice(), i-1}), 
                potential_.index({Slice(), i})
            ).clamp(logzero, 0);
    }
    // (B, T)
    auto partition = torch::logsumexp(chart, /*dim=*/{-1});

    return {chart, partition};
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
    double logzero = - std::numeric_limits<double>::infinity();
    /* pytorch syntax
    chart[:, T-1] = 0
    for t in reversed(range(0, T-1)):
        chart[:, t] = self.log_matmul(chart[:, t+1], potential[:, t+1])
    */
    int64_t start = time - 2;
    for (int64_t i = start; i > -1; --i) {
        chart.index({Slice(), i}) = 
            log_matmul(chart.index({Slice(), i+1}), 
                       potential.index({Slice(), i+1})
        ).clamp(logzero, 0);
    }

    return chart;
}

torch::Tensor hmm_fw_backward(
        const torch::Tensor& grad_z, 
        const torch::Tensor& potential, 
        const torch::Tensor& fwd_vars,
        const torch::Tensor& partition,
        const torch::Tensor& mask) {
    const int batch = potential.sizes()[0];
    const int time = potential.sizes()[1];
    const int N = potential.sizes()[2];
    torch::Tensor chart = torch::zeros({batch, time, N, N}, potential.device());
    double logzero = -1e23;
    // fwd and bwd vars
    torch::Tensor bwd_vars = hmm_bw_forward(potential);
    torch::Tensor bwd_vars_ = bwd_vars.unsqueeze(-2);
    torch::Tensor fwd_vars_ = fwd_vars.unsqueeze(-1);
    
    /*pytorch syntax
    chart = torch.zeros(B, T, self.num_codes, self.num_codes, device=partition.device)
    alpha_t = alpha_t.unsqueeze(-1)
    beta_t = beta_t.unsqueeze(-2)
    for t in range(0, T-1):
        chart[:, t] = alpha_t[:, t] + potential[:, t+1] + beta_t[:, t+1]
    marginal = chart - partition.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
    */ 
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
        chart.clamp(logzero, 0)
        - partition.view({batch, 1, 1, 1})
        + (-grad_z).log().view({batch, 1, 1, 1})
        + mask;

    return grad.exp();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &hmm_fw_forward, "HMM forward");
  m.def("backward", &hmm_fw_backward, "HMM backward");
}
