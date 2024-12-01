import torch
import torch.nn as nn
import torch.nn.functional as F

def heinsen_associative_scan_log(log_coeffs, log_values):
    """
    Implement the heinsen associative scan in log-space for numerical stability.
    This is the same implementation as in the original minGRU.
    """
    a_star = log_coeffs.cumsum(dim=1)
    log_h0_plus_b_star = (log_values - a_star).logcumsumexp(dim=1)
    log_h = a_star + log_h0_plus_b_star
    return log_h.exp()

def exists(val):
    return val is not None

# Simplified minGRU for stock prediction
class StockPredictorGRU(nn.Module):
    def __init__(self, input_dim, hidden_dim, expansion_factor=1.5):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # Like minGRU, use expansion factor for inner dimension
        dim_inner = int(hidden_dim * expansion_factor)
        
        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim, bias=False)
        
        # Combined hidden and gate computation like minGRU
        self.to_hidden_and_gate = nn.Linear(hidden_dim, dim_inner * 2, bias=False)
        
        # Output projection with expansion handling
        self.to_out = nn.Linear(dim_inner, hidden_dim, bias=False) if expansion_factor != 1. else nn.Identity()
        
        # Final projection for stock prediction
        self.final_proj = nn.Linear(hidden_dim, 1, bias=False)

    def g(self, x):
        return torch.where(x >= 0, x + 0.5, x.sigmoid())

    def log_g(self, x):
        return torch.where(x >= 0, (F.relu(x) + 0.5).log(), -F.softplus(-x))

    def forward(self, x, prev_hidden=None, return_next_prev_hidden=False):
        batch_size, seq_len, _ = x.shape

        # Project input to hidden dimension
        x = self.input_proj(x)
        hidden, gate = self.to_hidden_and_gate(x).chunk(2, dim=-1)

        if seq_len == 1:
            # Sequential processing
            hidden = self.g(hidden)
            gate = gate.sigmoid()
            out = torch.lerp(prev_hidden, hidden, gate) if exists(prev_hidden) else (hidden * gate)
        else:
            # Parallel processing
            log_coeffs = -F.softplus(gate)
            log_z = -F.softplus(-gate)
            log_tilde_h = self.log_g(hidden)
            log_values = log_z + log_tilde_h

            if exists(prev_hidden):
                log_values = torch.cat((self.log_g(prev_hidden), log_values), dim=1)
                log_coeffs = F.pad(log_coeffs, (0, 0, 1, 0))

            out = heinsen_associative_scan_log(log_coeffs, log_values)
            out = out[:, -seq_len:]

        next_prev_hidden = out[:, -1:]
        out = self.to_out(out)
        
        # Final projection for stock price
        predictions = self.final_proj(out)
        predictions = predictions.squeeze(-1)

        if not return_next_prev_hidden:
            return predictions

        return predictions, next_prev_hidden
