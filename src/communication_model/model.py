import torch
import torch.nn as nn


class TXEncoder(nn.Module):
    """
    Transformer-based transmitter.

    At each round t, TX receives:
      - original message symbols m
      - previous transmitted values prev_x
      - previous noisy received feedback values prev_y
      - current round index t

    It outputs one scalar coded symbol per message position.
    """

    def __init__(
        self,
        num_symbols: int = 8,
        message_len: int = 4,
        num_rounds: int = 4,
        d_symbol: int = 16,
        d_round: int = 8,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.num_symbols = num_symbols
        self.message_len = message_len
        self.num_rounds = num_rounds

        self.symbol_embedding = nn.Embedding(num_symbols, d_symbol)
        self.round_embedding = nn.Embedding(num_rounds, d_round)

        raw_dim = d_symbol + num_rounds + num_rounds + d_round

        self.pre_mlp = nn.Sequential(
            nn.Linear(raw_dim, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, message_len, d_model) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=False,
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )

        self.out_mlp = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
        )

    def forward(
        self,
        m: torch.Tensor,
        prev_x: torch.Tensor,
        prev_y: torch.Tensor,
        round_idx: int,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        m:
            Message tensor of shape (B, 4), values in {0,...,7}.

        prev_x:
            Previous transmitted history, shape (B, 4, T).

        prev_y:
            Previous noisy received feedback history, shape (B, 4, T).

        round_idx:
            Current round index, integer in {0,1,2,3}.

        Returns
        -------
        s_t:
            Raw unnormalized transmitted signal, shape (B, 4).
        """

        B = m.size(0)

        # Symbol embedding: (B, 4) -> (B, 4, d_symbol)
        e = self.symbol_embedding(m)

        # Round embedding: (B,) -> (B, d_round) -> (B, 4, d_round)
        r = torch.full(
            (B,),
            round_idx,
            dtype=torch.long,
            device=m.device,
        )
        r = self.round_embedding(r)
        r = r[:, None, :].expand(B, self.message_len, -1)

        # Concatenate token features:
        # [symbol embedding, previous x history, previous y feedback, round embedding]
        # Shape: (B, 4, d_symbol + T + T + d_round)
        u = torch.cat([e, prev_x, prev_y, r], dim=-1)

        # Map to transformer dimension
        z = self.pre_mlp(u)

        # Add positional embedding
        h = z + self.pos_embedding

        # Transformer over 4 message positions
        h = self.transformer(h)

        # One scalar coded symbol per position
        s_t = self.out_mlp(h).squeeze(-1)

        return s_t


class RXDecoder(nn.Module):
    """
    Transformer-based receiver.

    RX runs only after all T communication rounds.
    It receives Y_hist of shape (B, 4, T), where each message position has
    T noisy observations.
    """

    def __init__(
        self,
        num_symbols: int = 8,
        message_len: int = 4,
        num_rounds: int = 4,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.num_symbols = num_symbols
        self.message_len = message_len
        self.num_rounds = num_rounds

        self.pre_mlp = nn.Sequential(
            nn.Linear(num_rounds, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, message_len, d_model) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=False,
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )

        self.classifier = nn.Linear(d_model, num_symbols)

    def forward(self, y_hist: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        y_hist:
            Noisy received history, shape (B, 4, T).

        Returns
        -------
        logits:
            Class scores, shape (B, 4, 8).
        """

        # Map each length-T received vector to d_model
        z = self.pre_mlp(y_hist)

        # Add positional embedding
        h = z + self.pos_embedding

        # Transformer over 4 message positions
        h = self.transformer(h)

        # Predict one of 8 symbols for each position
        logits = self.classifier(h)

        return logits


class NeuralCommunicationSystem(nn.Module):
    """
    Full end-to-end communication system:
      TX -> power normalization -> AWGN channel -> relay feedback -> RX
    """

    def __init__(
        self,
        num_symbols: int = 8,
        message_len: int = 4,
        num_rounds: int = 4,
        sigma: float = 0.5,
        d_symbol: int = 16,
        d_round: int = 8,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.num_symbols = num_symbols
        self.message_len = message_len
        self.num_rounds = num_rounds
        self.sigma = sigma

        self.tx = TXEncoder(
            num_symbols=num_symbols,
            message_len=message_len,
            num_rounds=num_rounds,
            d_symbol=d_symbol,
            d_round=d_round,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )

        self.rx = RXDecoder(
            num_symbols=num_symbols,
            message_len=message_len,
            num_rounds=num_rounds,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )

    @staticmethod
    def power_normalize(s_t: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        """
        Enforces approximately:
            E ||x^(t)||_2^2 = 1

        s_t shape: (B, 4)
        """

        batch_power = s_t.pow(2).sum(dim=1).mean()
        x_t = s_t / torch.sqrt(batch_power + eps)

        return x_t

    def build_history(
        self,
        values: list,
        batch_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        """
        Converts a list of previous tensors into a padded history tensor.

        If we are at round t, values contains t previous tensors.
        Output shape is always (B, 4, T).

        This avoids unsafe in-place operations and preserves gradients through
        feedback from later rounds to earlier transmissions.
        """

        if len(values) == 0:
            return torch.zeros(
                batch_size,
                self.message_len,
                self.num_rounds,
                device=device,
            )

        hist = torch.stack(values, dim=-1)  # (B, 4, t)

        remaining = self.num_rounds - hist.size(-1)

        if remaining > 0:
            padding = torch.zeros(
                batch_size,
                self.message_len,
                remaining,
                device=device,
                dtype=hist.dtype,
            )
            hist = torch.cat([hist, padding], dim=-1)

        return hist

    def forward(self, m: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        m:
            Message tensor, shape (B, 4), values in {0,...,7}.

        Returns
        -------
        logits:
            Receiver output, shape (B, 4, 8).
        """

        B = m.size(0)
        device = m.device

        x_values = []
        y_values = []

        for t in range(self.num_rounds):
            # Build padded histories from previous rounds
            prev_x = self.build_history(x_values, B, device)
            prev_y = self.build_history(y_values, B, device)

            # TX generates raw coded symbols
            s_t = self.tx(m, prev_x, prev_y, round_idx=t)

            # Enforce average power constraint
            x_t = self.power_normalize(s_t)

            # AWGN channel
            noise = torch.randn_like(x_t) * self.sigma
            y_t = x_t + noise

            # Store for future feedback and final decoding
            x_values.append(x_t)
            y_values.append(y_t)

        # Final received history: (B, 4, T)
        y_hist = torch.stack(y_values, dim=-1)

        # RX decodes only at the end
        logits = self.rx(y_hist)

        return logits
