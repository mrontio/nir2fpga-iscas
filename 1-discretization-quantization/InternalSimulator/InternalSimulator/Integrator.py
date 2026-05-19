from __future__ import annotations

from typing import Any

import torch
import sinabs.layers as sl


class IntegratorSqueeze(sl.ExpLeakSqueeze):
    """Pure integrator: v[t] = v[t-1] + input[t]. No leak, no spiking."""

    def _forward_integrator(self, input_data: torch.Tensor) -> torch.Tensor:
        batch_size, time_steps, *trailing_dim = input_data.shape
        trailing_dim_tuple = tuple(trailing_dim)

        if not self.is_state_initialised():
            self.init_state_with_shape((batch_size, *trailing_dim_tuple))
        if not self.state_has_shape((batch_size, *trailing_dim_tuple)):
            if not self.has_trailing_dimension(trailing_dim_tuple):  # pyright: ignore[reportArgumentType]
                self.init_state_with_shape((batch_size, *trailing_dim_tuple))
            else:
                self.handle_state_batch_size_mismatch(batch_size)

        outputs: list[torch.Tensor] = []
        recordings: dict[str, list[torch.Tensor]] = {}
        if self.record_states:
            recordings = {"v_mem": []}

        for step in range(time_steps):
            self.v_mem = self.v_mem + input_data[:, step]
            outputs.append(self.v_mem.clone())
            if self.record_states:
                recordings["v_mem"].append(self.v_mem.clone())

        output = torch.stack(outputs, 1)
        self.recordings = (
            {name: torch.stack(vs, 1) for name, vs in recordings.items()}
            if self.record_states
            else {}
        )
        self.firing_rate = output.mean()
        return output

    def forward(self, input_data: torch.Tensor) -> Any:
        return self.squeeze_forward(input_data, self._forward_integrator)
