from __future__ import annotations

from typing import Any

import torch
import sinabs.layers as sl


class CurrentFilterSqueeze(sl.ExpLeakSqueeze):
    """Synaptic-current filter with CubaLIF-style dynamics.

    This differs from ``ExpLeakSqueeze`` by implementing
    ``i_syn[t+1] = alpha * (i_syn[t] + input[t])``.
    """

    def _forward_current_filter(self, input_data: torch.Tensor) -> torch.Tensor:
        batch_size, time_steps, *trailing_dim = input_data.shape
        trailing_dim_tuple = tuple(trailing_dim)

        if not self.is_state_initialised():
            self.init_state_with_shape((batch_size, *trailing_dim_tuple))

        if not self.state_has_shape((batch_size, *trailing_dim_tuple)):
            if not self.has_trailing_dimension(trailing_dim_tuple):  # pyright: ignore[reportArgumentType]
                self.init_state_with_shape((batch_size, *trailing_dim_tuple))
            else:
                self.handle_state_batch_size_mismatch(batch_size)

        alpha_mem = self.alpha_mem_calculated
        output_currents = []
        recordings: dict[str, list[torch.Tensor]] = {}
        if self.record_states:
            recordings = {"v_mem": []}

        for step in range(time_steps):
            self.v_mem = alpha_mem * (self.v_mem + input_data[:, step])
            output_currents.append(self.v_mem.clone())
            if self.record_states:
                recordings["v_mem"].append(self.v_mem.clone())

        output = torch.stack(output_currents, 1)
        if self.record_states:
            self.recordings = {
                name: torch.stack(values, 1) for name, values in recordings.items()
            }
        else:
            self.recordings = {}

        self.firing_rate = output.mean()
        return output

    def forward(self, input_data: torch.Tensor) -> Any:
        return self.squeeze_forward(input_data, self._forward_current_filter)
