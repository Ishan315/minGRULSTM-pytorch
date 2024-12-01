import torch
import torch.nn as nn
import torch.nn.functional as F

class minLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, bias=False):
        super(minLSTM, self).__init__()
        self.hidden_size = hidden_size

        # Linear layers for forget gate, input gate, and candidate hidden state
        self.linear_f = nn.Linear(input_size, hidden_size, bias=bias)
        self.linear_i = nn.Linear(input_size, hidden_size, bias=bias)
        self.linear_h = nn.Linear(input_size, hidden_size, bias=bias)

        # Activation functions
    def g(self, x):
        return torch.where(x >= 0, x + 0.5, x.sigmoid())

    def log_g(self, x):
        return torch.where(x >= 0, (F.relu(x) + 0.5).log(), -F.softplus(-x))

    def parallel_scan_log(self, log_coeffs, log_values):
        """
        Implements the parallel_scan_log function as described.

        Args:
            log_coeffs (Tensor): Logarithm of forget gates, shape (batch_size, seq_len, hidden_size).
            log_values (Tensor): Concatenated tensor of log_h0 and (log_i + log_tilde_h),
                                 shape (batch_size, seq_len + 1, hidden_size).

        Returns:
            Tensor: The hidden states after parallel scan, shape (batch_size, seq_len, hidden_size).
        """
        # Compute a_star: cumulative sum of log_coeffs along the sequence dimension (dim=1)
        # Shape: (batch_size, seq_len, hidden_size)
        a_star = torch.cumsum(log_coeffs, dim=1)

        # Pad a_star with a zero at the beginning of the sequence dimension
        # Shape after padding: (batch_size, seq_len + 1, hidden_size)
        a_star_padded = F.pad(a_star, (0, 0, 1, 0), "constant", 0)

        # Compute log_h0_plus_b_star using logcumsumexp for numerical stability
        # log_values has shape: (batch_size, seq_len + 1, hidden_size)
        # a_star_padded has shape: (batch_size, seq_len + 1, hidden_size)
        # log_values - a_star_padded computes element-wise subtraction
        # logcumsumexp is applied along the sequence dimension (dim=1)
        log_h0_plus_b_star = torch.logcumsumexp(log_values - a_star_padded, dim=1)
        # Compute log_h by adding a_star_padded and log_h0_plus_b_star
        log_h = a_star_padded + log_h0_plus_b_star  # Shape: (batch_size, seq_len +1, hidden_size)

        # Remove the first element corresponding to the padding
        # Shape: (batch_size, seq_len, hidden_size)
        log_h = log_h[:, 1:, :]

        # Exponentiate to get h in the original space
        h = torch.exp(log_h)

        return h  # Shape: (batch_size, seq_len, hidden_size)
    
    def forward(self, x, h_0):
        """
        Forward pass of the minLSTM.

        Args:
            x (Tensor): Input tensor of shape (batch_size, seq_len, input_size).
            h_0 (Tensor): Initial hidden state of shape (batch_size, hidden_size).

        Returns:
            Tensor: The hidden state after processing, shape depends on the mode:
                    - Sequential Mode: (batch_size, hidden_size)
                    - Parallel Mode: (batch_size, seq_len, hidden_size)
        """
        batch_size, seq_len, input_size = x.size()

        if seq_len == 1:
            # === Sequential Mode ===
            # Extract the single time step
            x_t = x[:, 0, :]  # Shape: (batch_size, input_size)
            h_prev = h_0      # Shape: (batch_size, hidden_size)

            # Compute gates
            f_t = torch.sigmoid(self.linear_f(x_t))  # Forget gate
            i_t = torch.sigmoid(self.linear_i(x_t))  # Input gate

            # Compute candidate hidden state
            tilde_h_t = self.g(self.linear_h(x_t))   # Candidate hidden state

            # Normalize gates
            f_prime_t = f_t / (f_t + i_t)
            i_prime_t = i_t / (f_t + i_t)

            # Update hidden state
            h_t = f_prime_t * h_prev + i_prime_t * tilde_h_t  # Shape: (batch_size, hidden_size)

            return h_t  # Sequential output
        else:
            # === Parallel Mode ===
            # Compute log-space forget and input gates
            log_f = -F.softplus(-self.linear_f(x))  # Shape: (batch_size, seq_len, hidden_size)
            log_i = -F.softplus(-self.linear_i(x))  # Shape: (batch_size, seq_len, hidden_size)

            # Compute log-space candidate hidden state
            log_tilde_h = self.log_g(self.linear_h(x))  # Shape: (batch_size, seq_len, hidden_size)

            # Compute difference for gate normalization
            diff = F.softplus(-self.linear_f(x)) - F.softplus(-self.linear_i(x))  # Shape: (batch_size, seq_len, hidden_size)

            # Update log gates based on the difference
            log_f = -F.softplus(diff)      # Updated forget gate
            log_i = -F.softplus(-diff)     # Updated input gate

            # Compute log-space initial hidden state
            log_h0 = self.log_g(h_0)       # Shape: (batch_size, hidden_size)

            # Prepare concatenated tensor for parallel scan
            log_h0 = log_h0.unsqueeze(1)    # Shape: (batch_size, 1, hidden_size)
            log_i_plus_log_tilde_h = log_i + log_tilde_h  # Shape: (batch_size, seq_len, hidden_size)
            concatenated = torch.cat([log_h0, log_i_plus_log_tilde_h], dim=1)  # Shape: (batch_size, seq_len + 1, hidden_size)

            # Perform parallel scan
            h = self.parallel_scan_log(log_f, concatenated)  # Shape: (batch_size, seq_len, hidden_size)

            return h  # Parallel output

class StockPredictorLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim, bias=False)
        
        # minLSTM layer
        self.lstm = minLSTM(hidden_dim, hidden_dim)
        
        # Final projection for stock prediction
        self.final_proj = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, x, h_0=None):
        batch_size, seq_len, _ = x.shape
        
        # Initialize h_0 if not provided
        if h_0 is None:
            h_0 = torch.zeros(batch_size, self.hidden_dim, device=x.device, dtype=x.dtype)

        # Project input to hidden dimension
        x = self.input_proj(x)
        
        # Process through minLSTM
        hidden_states = self.lstm(x, h_0)
        
        # Final projection for stock price
        predictions = self.final_proj(hidden_states)
        predictions = predictions.squeeze(-1)

        return predictions