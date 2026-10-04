### TODO

- **Blocked matmul for 64 action rows (S64)**
  - With an action chunk of 33-64 actions, the expert processes two 32-row tiles of action rows.
  - Today each expert matmul (qkv, o_proj, up/gate, down) is called once for each row tile. So each weight tile is unpacked twice.
  - Plan: call each matmul once for both row tiles (`rt_dim=2`). Each weight tile is then unpacked once. The output is expected to stay bit-identical.
  - Measured ceiling: removing the second row tile's matmuls saves 10.5 us per layer (qkv 2.3, o_proj 2.4, MLP 5.7) at 2 cameras / 224-token bucket / H = 50. That is at most about 0.19 ms per denoising step, or 1.9 ms per call at N = 10. The real gain will be smaller.
  - Open risk: a blocked call uses twice the DST tiles. Some matmuls may need a new DST split.
- **Expert fidelity per preset**
  - HiFi4 on the expert fixes the two A4 cells at H = 1, but its speed changes with the preset (-2.5 to +1.6 ms). Choose the fidelity per preset class.
