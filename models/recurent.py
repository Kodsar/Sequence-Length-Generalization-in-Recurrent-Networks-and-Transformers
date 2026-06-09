from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn


class RecurrentCell(nn.Module):

    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # One linear for x and one for h per gate (clear + stable)
        self.W_xz = nn.Linear(input_dim, hidden_dim, bias=True)
        self.W_hz = nn.Linear(hidden_dim, hidden_dim, bias=False)

        self.W_xr = nn.Linear(input_dim, hidden_dim, bias=True)
        self.W_hr = nn.Linear(hidden_dim, hidden_dim, bias=False)

        self.W_xn = nn.Linear(input_dim, hidden_dim, bias=True)
        self.W_hn = nn.Linear(hidden_dim, hidden_dim, bias=False)

    def forward(self, x: torch.Tensor, h_prev: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x:      (B, input_dim)
            h_prev: (B, hidden_dim)
        Returns:
            h_new:  (B, hidden_dim)
        """
        z = torch.sigmoid(self.W_xz(x) + self.W_hz(h_prev))
        r = torch.sigmoid(self.W_xr(x) + self.W_hr(h_prev))
        n = torch.tanh(self.W_xn(x) + self.W_hn(r * h_prev))
        h_new = (1.0 - z) * n + z * h_prev
        return h_new


class RecurrentEncoder(nn.Module):
    """
    Embedding + stacked GRU cells (+ optional bidirectional) with dropout between layers.

    Key points:
    - No nn.utils.rnn, no nn.GRU/LSTM/RNN modules.
    - "Effective batch size" trick: sort by length descending and for time t
      only run on prefix b_eff samples where length > t.
    - Implementation is MPS-safe: avoids in-place ops that break autograd.
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        hidden_dim: int,
        num_layers: int,
        pad_token: int,
        bidirectional: bool = False,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.pad_token = pad_token
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.dropout_p = dropout

        # Important: use vocab_size + 1 so it's safe for both tasks:
        # - modular_addition: vocab includes PAD already (0..30 + 31 pad) => VOCAB_SIZE=32, PAD=31
        # - palindrome: vocab=10, PAD=10 => need size 11
        self.embedding = nn.Embedding(vocab_size + 1, embed_dim, padding_idx=pad_token)

        self.dropout = nn.Dropout(dropout)

        # Unidirectional stacks for fwd/bwd
        self.fw_cells = nn.ModuleList()
        self.bw_cells = nn.ModuleList() if bidirectional else None

        in_dim = embed_dim
        for _ in range(num_layers):
            self.fw_cells.append(RecurrentCell(in_dim, hidden_dim))
            if bidirectional:
                self.bw_cells.append(RecurrentCell(in_dim, hidden_dim))
            in_dim = (2 * hidden_dim) if bidirectional else hidden_dim  # per direction stack stays H (concat handled outside)

    @staticmethod
    def _sort_by_length(x: torch.Tensor, lengths: torch.Tensor):
        lengths_sorted, sort_idx = torch.sort(lengths, descending=True)
        x_sorted = x.index_select(0, sort_idx)
        inv_idx = torch.empty_like(sort_idx)
        inv_idx[sort_idx] = torch.arange(sort_idx.size(0), device=sort_idx.device)
        return x_sorted, lengths_sorted, sort_idx, inv_idx

    def _reverse_valid_prefix(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """
        Reverse only the valid (non-pad) prefix of each sequence.
        x: (B, T, D)
        """
        B, T, D = x.shape
        x_rev = x.clone()
        for i in range(B):
            L = int(lengths[i].item())
            if L > 1:
                x_rev[i, :L] = torch.flip(x[i, :L], dims=[0])
        return x_rev

    def _run_stack_unidirectional(
        self,
        x_emb: torch.Tensor,         # (B, T, D_in)
        lengths: torch.Tensor,       # (B,)
        cell_stack: nn.ModuleList,   # L layers
    ) -> torch.Tensor:
        """
        Runs a multi-layer GRU forward (unidirectional).

        Returns:
            out: (B, T, H)  (last layer hidden per time step)
        """
        B, T, _ = x_emb.shape
        device = x_emb.device
        dtype = x_emb.dtype

        layer_input = x_emb
        for layer_idx, cell in enumerate(cell_stack):
            h = torch.zeros(B, self.hidden_dim, device=device, dtype=dtype)
            outputs = torch.zeros(B, T, self.hidden_dim, device=device, dtype=dtype)

            for t in range(T):
                b_eff = int((lengths > t).sum().item())
                if b_eff <= 0:
                    break

                # compute only for the active prefix
                h_eff = h[:b_eff]
                x_eff = layer_input[:b_eff, t, :]
                h_new = cell(x_eff, h_eff)

                # MPS-safe update: clone then write prefix
                h_new_full = h.clone()
                h_new_full[:b_eff] = h_new
                h = h_new_full

                # outputs buffer can be written (it is a fresh tensor), but we keep it safe too
                outputs[:b_eff, t, :] = h_new

            if layer_idx != len(cell_stack) - 1 and self.dropout_p > 0:
                outputs = self.dropout(outputs)

            layer_input = outputs

        return layer_input  # (B, T, H)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T) token ids with pad_token
            lengths: (B,) true lengths
        Returns:
            if bidirectional: (B, T, 2H)
            else: (B, T, H)
        """
        x_sorted, lengths_sorted, _, inv_idx = self._sort_by_length(x, lengths)
        x_emb = self.embedding(x_sorted)  # (B, T, E)

        if not self.bidirectional:
            out = self._run_stack_unidirectional(x_emb, lengths_sorted, self.fw_cells)
            return out.index_select(0, inv_idx)

        # Bidirectional:
        # For each layer, we run forward and backward on that layer's input, then concat.
        layer_input = x_emb
        for layer_idx in range(self.num_layers):
            fw_cell = self.fw_cells[layer_idx]
            bw_cell = self.bw_cells[layer_idx]

            # forward for this layer (single-layer run)
            fw_out = self._run_stack_unidirectional(layer_input, lengths_sorted, nn.ModuleList([fw_cell]))

            # backward: reverse valid prefix, run forward, then reverse outputs back
            rev_in = self._reverse_valid_prefix(layer_input, lengths_sorted)
            bw_out_rev = self._run_stack_unidirectional(rev_in, lengths_sorted, nn.ModuleList([bw_cell]))
            bw_out = self._reverse_valid_prefix(bw_out_rev, lengths_sorted)

            out = torch.cat([fw_out, bw_out], dim=-1)  # (B, T, 2H)

            if layer_idx != self.num_layers - 1 and self.dropout_p > 0:
                out = self.dropout(out)

            # next layer input is concatenated representation
            layer_input = out

            # IMPORTANT: next layer cells expect input_dim = H, not 2H, in this implementation.
            # If you want true bidirectional stacking (where next layer sees 2H),
            # then you must build cells with input_dim=2H for deeper layers.
            # In this project configs: palindrome uses bidirectional=True, modular_addition uses False.
            # So for palindrome (num_layers small) this is fine; for deep bi stacks you can extend it.

        return layer_input.index_select(0, inv_idx)


class RecurrentClassifier(nn.Module):
    """
    Encoder + Linear head.

    For modular_addition training:
    - return_last_step_only=False => (B,T,C) for deep supervision
    For palindrome classification:
    - return_last_step_only=True  => (B,C)
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        hidden_dim: int,
        num_layers: int,
        num_classes: int,
        pad_token: int,
        bidirectional: bool = False,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.bidirectional = bidirectional
        self.hidden_dim = hidden_dim

        self.encoder = RecurrentEncoder(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            pad_token=pad_token,
            bidirectional=bidirectional,
            dropout=dropout,
        )

        out_dim = (2 * hidden_dim) if bidirectional else hidden_dim
        self.classifier = nn.Linear(out_dim, num_classes)

    def forward(
        self,
        x: torch.Tensor,
        lengths: torch.Tensor,
        return_last_step_only: bool = True,
    ) -> torch.Tensor:
        enc = self.encoder(x, lengths)  # (B, T, H) or (B, T, 2H)

        if not return_last_step_only:
            return self.classifier(enc)  # (B, T, C)

        B = enc.size(0)
        batch_idx = torch.arange(B, device=enc.device)
        last_idx = lengths - 1

        if not self.bidirectional:
            last_h = enc[batch_idx, last_idx, :]  # (B, H)
            return self.classifier(last_h)        # (B, C)

        # bidirectional: enc = [fw | bw]
        fw = enc[..., : self.hidden_dim]
        bw = enc[..., self.hidden_dim :]

        fw_last = fw[batch_idx, last_idx, :]  # (B, H)
        bw_first = bw[batch_idx, 0, :]        # (B, H)

        seq_repr = torch.cat([fw_last, bw_first], dim=-1)  # (B, 2H)
        return self.classifier(seq_repr)                   # (B, C)
