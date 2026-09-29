# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Prefix / expert DISAGGREGATED pi0.5 on a 2x p300 box (v2 step 2; the server's ``multi-robot`` profile).

One 1x4 parent mesh (the fabric needs the whole Ethernet ring) is carved into two sub-meshes:

* **A** (chips 0..1, one p300): the prefix -- host im2col -> SigLIP -> Gemma-2B VLM prefill, tensor-parallel
  over its two chips (``FusedConfig.tp`` = 2) -- writing the 18 layers' K/V of B requests into caches that
  are then SENT over mesh sockets (``ttnn.experimental.send_async``) to B.
* **B** (chips 2..3, the other p300): the Gemma-300M action expert, replicated on both chips, RECEIVING the
  K/V into a cache set and running the 10 denoising steps on the B requests.

The two stages run concurrently: while B denoises request set n, A prefills request set n+1. Each stage is
one Metal trace (the transfers are inside the traces when the runtime allows it, else issued eagerly right
after / before them). Expected per-request throughput ~ 1 / max(T_prefix(A), T_expert(B)).

Several batch sizes can be prepared at once (``batches=(1, 2, 4)``): every batch size gets its own persistent
inputs, K/V cache set(s) and traces on both boards, so a server can batch dynamically (the pipeline is fed
one request set at a time through ``write_prefix_inputs`` / ``launch_prefix`` / ``launch_expert`` /
``read_expert``; ``run_one`` and ``run_pipelined`` are the benchmark entry points).
"""

from __future__ import annotations

import time
from dataclasses import replace
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import ttnn

from models.experimental.pi0_5.common.configs import PI0ModelConfig
from models.experimental.pi0_5.common.fused_config import FusedConfig
from models.experimental.pi0_5.common.fused_host import check_fused_shape_contract, euler_dts, pad_rows, unpad_rows
from models.experimental.pi0_5.common.weight_loader import PI0WeightLoader
from models.experimental.pi0_5.tt.ttnn_pi0_model import PI0ModelTTNN


def build_socket_connections(mesh_shape, num_connections: int, sender_row: int = 0, receiver_row: int = 1, mirror: bool = True):
    """One connection per (mesh coordinate, worker core): sender core (i, sender_row) on A's chip -> receiver core
    (i, receiver_row) on B's chip. With ``mirror`` A's local column c talks to B's local column cols-1-c: on the
    2x p300 ring (parent 1x4 opened as chips [1,0,3,2]) the Ethernet neighbours are A0<->B1 and A1<->B0, which
    is also what the 1D-line fabric requires (sender and receiver in the same global row or column). The K/V are
    replicated on A's chips, so any bijection is correct."""
    rows, cols = mesh_shape[0], mesh_shape[1]
    connections = []
    for coord in ttnn.MeshCoordinateRange(mesh_shape):
        r, c = coord[0], coord[1]
        recv_coord = ttnn.MeshCoordinate(r, cols - 1 - c) if mirror else coord
        for i in range(num_connections):
            connections.append(
                ttnn.SocketConnection(
                    ttnn.MeshCoreCoord(coord, ttnn.CoreCoord(i, sender_row)),
                    ttnn.MeshCoreCoord(recv_coord, ttnn.CoreCoord(i, receiver_row)),
                )
            )
    return connections


class PI05DisaggPipeline:
    def __init__(
        self,
        parent_mesh,
        config: PI0ModelConfig,
        loader: PI0WeightLoader,
        fused_cfg: FusedConfig,
        batch: int = 1,
        batches: Optional[Sequence[int]] = None,  # several prepared batch sizes (server); default (batch,)
        num_images: int = 2,
        token_len: int = 224,
        prefix_chips: int = 2,
        socket_connections: int = 2,  # = the 2 fabric links between the boards (7.7 MB in 0.49 ms; 1 link 0.86 ms; 4 is refused)
        socket_fifo_bytes: int = 128 * 1024,  # 512K clashed with the fused attention CBs at batch 2/4 (same throughput)
        transfer_in_trace: bool = True,
        trace: bool = True,
        transfer_mode: str = "async",  # "async" (FIFO send_async / recv_async) | "direct" (send_direct_async / recv_direct_async)
        recv_ahead: bool = False,  # B's trace for set n receives set n+1's K/V between its layers (stalls the compute stream: slow)
        recv_cq1: bool = False,  # EXPERIMENTAL (hangs on this runtime, 2026-09-17): the 36 recvs of a set as their own trace on
                                 # B's SECOND command queue, overlapping the previous set's expert on CQ0 (events CQ1 -> CQ0)
    ):
        self.parent = parent_mesh
        self.config = config
        self.batches: Tuple[int, ...] = tuple(sorted({int(b) for b in (batches if batches else (batch,))}))
        if not self.batches or self.batches[0] < 1:
            raise ValueError(f"batch sizes must be positive: {self.batches}")
        self.batch = self.batches[0]  # the benchmark scripts' single batch size
        self.num_images = num_images
        self.token_len = token_len
        self.trace = trace
        n_parent = parent_mesh.get_num_devices()
        assert 0 < prefix_chips < n_parent, (prefix_chips, n_parent)
        self.A = parent_mesh.create_submesh(ttnn.MeshShape(1, prefix_chips), ttnn.MeshCoordinate(0, 0))
        self.B = parent_mesh.create_submesh(ttnn.MeshShape(1, n_parent - prefix_chips), ttnn.MeshCoordinate(0, prefix_chips))
        self.A.enable_program_cache()
        self.B.enable_program_cache()
        cfg_a = fused_cfg.resolved(self.A.get_num_devices())  # prefix tensor-parallel over A
        cfg_b = replace(fused_cfg, tp=1).resolved(self.B.get_num_devices())  # expert replicated on B
        cfg_b = replace(cfg_b, tp=1)
        t0 = time.perf_counter()
        self.prefix_model = PI0ModelTTNN(config, loader, self.A, fused=cfg_a)
        self.expert_model = PI0ModelTTNN(config, loader, self.B, fused=cfg_b)
        self.build_seconds = time.perf_counter() - t0
        self.transfer_in_trace = transfer_in_trace
        self.transfer_mode = transfer_mode
        self.recv_cq1 = recv_cq1 and trace
        self.recv_ahead = recv_ahead and transfer_in_trace and not self.recv_cq1
        if self.recv_cq1:
            self.transfer_in_trace = False  # A: sends stay in A's trace (below); B: recvs live in the CQ1 traces
        # B's trace receives its set's K/V right before computing on it, so one cache set per batch size is enough;
        # the recv-ahead / CQ1 variants receive set n+1 while set n computes and need ping / pong.
        self.num_sets = 2 if (self.recv_ahead or self.recv_cq1) else 1
        # Per batch size: shape plan, A's cache set (owned by the prefix backbone's per-batch sets) and B's set(s).
        self.plan: Dict[int, Dict[str, int]] = {}
        self.kv_plan: Dict[int, Dict[str, int]] = {}
        self.cache_sets: Dict[int, List[list]] = {}
        for b in self.batches:
            plan = check_fused_shape_contract(num_images, token_len, config.action_horizon, batch=b)
            self.plan[b] = plan
            self.prefix_model.backbone.allocate_kv_caches(plan["prefix_len"], config.action_horizon, batch=b)
            self.kv_plan[b] = self.expert_model.backbone.allocate_kv_caches(plan["prefix_len"], config.action_horizon, batch=b)
            set0 = self.expert_model.backbone.kv_caches
            sets = [set0]
            for _ in range(1, self.num_sets):
                sets.append(
                    [
                        (ttnn.zeros(k.shape, dtype=k.dtype, layout=k.layout, device=self.B, memory_config=k.memory_config()),
                         ttnn.zeros(v.shape, dtype=v.dtype, layout=v.layout, device=self.B, memory_config=v.memory_config()))
                        for k, v in set0
                    ]
                )
            self.cache_sets[b] = sets
        # sockets A -> B (one pair; the 36 K/V tensors of a set go through it in a fixed order)
        mem = ttnn.SocketMemoryConfig(ttnn.BufferType.L1, socket_fifo_bytes)
        self.socket_cfg = ttnn.SocketConfig(build_socket_connections(self.A.shape, socket_connections), mem)
        self.send_socket, self.recv_socket = ttnn.create_socket_pair(self.A, self.B, self.socket_cfg)
        if transfer_mode == "direct":
            self._send_op, self._recv_op = ttnn.experimental.send_direct_async, ttnn.experimental.recv_direct_async
        else:
            self._send_op, self._recv_op = ttnn.experimental.send_async, ttnn.experimental.recv_async
        # per batch size: persistent inputs, traces, outputs
        self._in: Dict[int, Dict[str, ttnn.Tensor]] = {}
        self._trace_a: Dict[int, Optional[int]] = {b: None for b in self.batches}
        self._trace_b: Dict[int, List] = {b: [None] * self.num_sets for b in self.batches}
        self._out_b: Dict[int, List] = {b: [None] * self.num_sets for b in self.batches}
        self._recv_trace: Dict[int, List] = {b: [None] * self.num_sets for b in self.batches}
        self._recv_event: Dict[int, List] = {b: [None] * self.num_sets for b in self.batches}
        self._sends_in_trace_a = self.transfer_in_trace or self.recv_cq1
        self.timing: Dict[str, float] = {}

    # ------------------------------------------------------------------ caches
    def _a_caches(self, b: int):
        """A's K/V set for batch ``b`` (switches the prefix backbone's active set; no device work once allocated)."""
        pm = self.prefix_model.backbone
        pm.allocate_kv_caches(self.plan[b]["prefix_len"], self.config.action_horizon, batch=b)
        return pm.kv_caches

    def _bind_expert_caches(self, b: int, set_idx: int) -> None:
        em = self.expert_model.backbone
        em.kv_caches = self.cache_sets[b][set_idx]
        em.kv_cache_plan = self.kv_plan[b]

    # ------------------------------------------------------------------ host inputs
    def host_inputs(self, images: List[torch.Tensor], tokens: torch.Tensor, noise: Optional[torch.Tensor] = None):
        """torch request set -> (im2col host, tokens host, noise host, batch). ``batch`` = tokens.shape[0] must be one
        of the prepared batch sizes."""
        b = int(tokens.shape[0]) if tokens.dim() > 1 else 1
        if b not in self.batches:
            raise ValueError(f"batch {b} is not prepared; prepared batch sizes: {self.batches}")
        im2col_hosts, tokens_host, _ = self.prefix_model._fused_host_inputs(images, tokens, None)
        assert len(im2col_hosts) == 1, "disaggregated path needs PI05_SIGLIP_BATCHED=1"
        noise_t = self.expert_model._default_noise_torch if noise is None else noise
        noise_t = noise_t.reshape(-1, self.config.action_horizon, self.config.action_dim).float()
        if noise_t.shape[0] == 1 and b > 1:
            noise_t = noise_t.expand(b, -1, -1).contiguous()
        if noise_t.shape[0] != b:
            raise ValueError(f"noise batch {noise_t.shape[0]} != {b} requests")
        noise_host = ttnn.from_torch(
            pad_rows(noise_t, self.expert_model._suffix_rows), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.B),
        )
        return im2col_hosts[0], tokens_host, noise_host, b

    # ------------------------------------------------------------------ device graphs
    def _graph_a(self, b: int, with_transfer: bool):
        pm = self.prefix_model
        self._a_caches(b)  # forward_vlm_fused writes the active set
        inp = self._in[b]
        prefix_embs = pm.prefix_embedding.embed_prefix_fused([inp["im2col"]], inp["tokens"])
        pm.backbone.forward_vlm_fused(prefix_embs)
        if with_transfer:
            self._send_caches(b)

    def _send_caches(self, b: int):
        for k, v in self._a_caches(b):
            self._send_op(k, self.send_socket)
            self._send_op(v, self.send_socket)

    def _recv_layer(self, b: int, set_idx: int, layer: int):
        k, v = self.cache_sets[b][set_idx][layer]
        self._recv_op(k, self.recv_socket)
        self._recv_op(v, self.recv_socket)

    def _recv_caches(self, b: int, set_idx: int):
        for layer in range(len(self.cache_sets[b][set_idx])):
            self._recv_layer(b, set_idx, layer)

    def _forward_expert_with_prefetch(self, em, hidden, block_mods, final_mod, b: int, prefetch_set: Optional[int]):
        """One denoising step; with ``prefetch_set`` the K/V of the NEXT request set are received (layer by
        layer, in A's send order) between this step's layers, so the transfer hides behind the compute."""
        prefix_len = em.backbone.kv_cache_plan["prefix_len"]
        for i, block in enumerate(em.backbone.expert_blocks):
            cache_k, cache_v = em.backbone.kv_caches[i]
            hidden = block.forward_fused_expert(hidden, block_mods[i], cache_k, cache_v, prefix_len)
            if prefetch_set is not None:
                self._recv_layer(b, prefetch_set, i)
        from models.experimental.pi0_5.tt.ttnn_gemma import adarms_norm_precomputed

        out = adarms_norm_precomputed(hidden, final_mod[0], final_mod[1], em.config.expert_config.rms_norm_eps)
        ttnn.deallocate(hidden)
        return out

    def _graph_b(self, b: int, set_idx: int, with_transfer: bool) -> ttnn.Tensor:
        em = self.expert_model
        prefetch = None
        if with_transfer and self.recv_ahead:
            prefetch = 1 - set_idx  # this set's own K/V arrived during the PREVIOUS trace (or the priming recv)
        elif with_transfer:
            self._recv_caches(b, set_idx)
        self._bind_expert_caches(b, set_idx)
        noise_in = self._in[b]["noise"]
        x_t = noise_in
        for i, dt in enumerate(euler_dts(em.denoise_config.num_steps)):
            suffix = em.suffix_embedding.embed_actions_fused(x_t)
            if prefetch is not None and i == 0:
                out = self._forward_expert_with_prefetch(
                    em, suffix, em._precomputed_block_mods[i], em._precomputed_final_mod[i], b, prefetch
                )
            else:
                out = em.backbone.forward_expert_fused(suffix, em._precomputed_block_mods[i], em._precomputed_final_mod[i])
            x_next = em.suffix_embedding.euler_step_fused(out, x_t, dt)
            ttnn.deallocate(out)
            if x_t is not noise_in:
                ttnn.deallocate(x_t)
            x_t = x_next
        return x_t

    # ------------------------------------------------------------------ prepare (compile + capture)
    def prepare(self, images, tokens, noise=None):
        """Compile and capture EVERY prepared batch size from one request set's inputs (a set of n requests is
        tiled to each batch size, so one warm-up request prepares them all)."""
        n = int(tokens.shape[0]) if tokens.dim() > 1 else 1
        t0 = time.perf_counter()
        for b in self.batches:
            if b % n != 0:
                raise ValueError(f"cannot tile {n} warm-up requests to batch {b}")
            rep = b // n
            images_b = list(images) * rep
            tokens_b = tokens.reshape(n, -1).repeat(rep, 1)
            noise_b = None
            if noise is not None:
                noise_b = noise.reshape(-1, self.config.action_horizon, self.config.action_dim)
                noise_b = (noise_b.expand(n, -1, -1) if noise_b.shape[0] == 1 else noise_b).repeat(rep, 1, 1)
            im2col_host, tokens_host, noise_host, _ = self.host_inputs(images_b, tokens_b, noise_b)
            self._in[b] = dict(
                im2col=ttnn.to_device(im2col_host, self.A, memory_config=ttnn.DRAM_MEMORY_CONFIG),
                tokens=ttnn.to_device(tokens_host, self.A, memory_config=ttnn.DRAM_MEMORY_CONFIG),
                noise=ttnn.to_device(noise_host, self.B, memory_config=ttnn.L1_MEMORY_CONFIG),
            )
            # eager compile pass: A prefill + sends, B recvs + expert (every cache set)
            self._graph_a(b, with_transfer=False)
            self._send_caches(b)
            for s in range(self.num_sets):
                self._recv_caches(b, s)
                out = self._graph_b(b, s, with_transfer=False)
                ttnn.synchronize_device(self.B)
                ttnn.deallocate(out)
                if s + 1 < self.num_sets:
                    self._send_caches(b)
            ttnn.synchronize_device(self.A)
            ttnn.synchronize_device(self.B)
        self.timing["compile_ms"] = (time.perf_counter() - t0) * 1000
        if not self.trace:
            return
        t0 = time.perf_counter()
        for b in self.batches:
            # trace A (prefill + sends; the sends block A's queue until B receives -- harmless, A is done computing)
            tid = ttnn.begin_trace_capture(self.A, cq_id=0)
            try:
                self._graph_a(b, with_transfer=self._sends_in_trace_a)
            finally:
                ttnn.end_trace_capture(self.A, tid, cq_id=0)
            self._trace_a[b] = tid
            if self.recv_cq1:
                # recv traces on CQ1 (one per cache set): the eager compile pass above already ran the recv programs
                for s in range(self.num_sets):
                    tid_r = ttnn.begin_trace_capture(self.B, cq_id=1)
                    try:
                        self._recv_caches(b, s)
                    finally:
                        ttnn.end_trace_capture(self.B, tid_r, cq_id=1)
                    self._recv_trace[b][s] = tid_r
            # traces B: [recvs +] expert. Capture records the socket ops without running them (the eager compile
            # pass above already built their programs); at run time A's trace (sends) and B's trace (recvs) execute
            # concurrently and pair up through the socket FIFO.
            for s in range(self.num_sets):
                tid_b = ttnn.begin_trace_capture(self.B, cq_id=0)
                try:
                    self._out_b[b][s] = self._graph_b(b, s, with_transfer=self.transfer_in_trace)
                finally:
                    ttnn.end_trace_capture(self.B, tid_b, cq_id=0)
                self._trace_b[b][s] = tid_b
        ttnn.synchronize_device(self.A)
        ttnn.synchronize_device(self.B)
        self.timing["capture_ms"] = (time.perf_counter() - t0) * 1000

    # ------------------------------------------------------------------ run (stage API)
    def write_prefix_inputs(self, b: int, im2col_host, tokens_host) -> None:
        ttnn.copy_host_to_device_tensor(im2col_host, self._in[b]["im2col"], cq_id=0)
        ttnn.copy_host_to_device_tensor(tokens_host, self._in[b]["tokens"], cq_id=0)

    def write_expert_inputs(self, b: int, noise_host) -> None:
        ttnn.copy_host_to_device_tensor(noise_host, self._in[b]["noise"], cq_id=0)

    def launch_prefix(self, b: int) -> None:
        """Enqueue A's prefill (+ the K/V sends) for batch ``b``; returns at once."""
        if self._trace_a[b] is not None:
            ttnn.execute_trace(self.A, self._trace_a[b], cq_id=0, blocking=False)
            if not self._sends_in_trace_a:
                self._send_caches(b)
        else:
            self._graph_a(b, with_transfer=True)

    def _launch_recv(self, b: int, set_idx: int):
        """CQ1 mode: enqueue the recv trace of ``set_idx`` on B's second queue and record its completion event."""
        ttnn.execute_trace(self.B, self._recv_trace[b][set_idx], cq_id=1, blocking=False)
        self._recv_event[b][set_idx] = ttnn.record_event(self.B, cq_id=1)

    def launch_expert(self, b: int, set_idx: int = 0) -> None:
        """Enqueue B's K/V receive + 10 denoising steps for batch ``b`` (pairs with the previous launch_prefix(b))."""
        if self._trace_b[b][set_idx] is not None:
            if self.recv_cq1:
                if self._recv_event[b][set_idx] is None:
                    self._launch_recv(b, set_idx)
                ttnn.wait_for_event(0, self._recv_event[b][set_idx])  # CQ0 waits for this set's K/V
                self._recv_event[b][set_idx] = None
            elif not self.transfer_in_trace:
                self._recv_caches(b, set_idx)
            ttnn.execute_trace(self.B, self._trace_b[b][set_idx], cq_id=0, blocking=False)
        else:
            self._out_b[b][set_idx] = self._graph_b(b, set_idx, with_transfer=True)

    def read_expert(self, b: int, set_idx: int = 0) -> torch.Tensor:
        """Block until B is done and return the ``[b, 50, 32]`` actions (chip 0 of B; the expert is replicated)."""
        out = self._out_b[b][set_idx]
        shard = ttnn.get_device_tensors(out)[0] if self.B.get_num_devices() > 1 else out
        actions = ttnn.to_torch(shard).float()
        return unpad_rows(actions, self.config.action_horizon)

    def run_one(self, images, tokens, noise=None, set_idx: int = 0) -> torch.Tensor:
        """Un-pipelined single request set: A then B, returns [B, 50, 32] actions."""
        im2col_host, tokens_host, noise_host, b = self.host_inputs(images, tokens, noise)
        self.write_prefix_inputs(b, im2col_host, tokens_host)
        self.write_expert_inputs(b, noise_host)
        self.launch_prefix(b)
        if self.recv_ahead:
            self._recv_caches(b, set_idx)  # this set's data (eager)
            self.launch_prefix(b)  # the trace's embedded prefetch of the other set needs a sender
        if self.recv_cq1:
            self._launch_recv(b, set_idx)
        self.launch_expert(b, set_idx)
        return self.read_expert(b, set_idx)

    # backwards-compatible names used by the step-2 benchmarks
    def _launch_a(self, b: Optional[int] = None):
        self.launch_prefix(self.batch if b is None else b)

    def _launch_b(self, set_idx: int = 0, b: Optional[int] = None):
        self.launch_expert(self.batch if b is None else b, set_idx)

    def _read_b(self, set_idx: int = 0, b: Optional[int] = None):
        return self.read_expert(self.batch if b is None else b, set_idx)

    def run_pipelined(self, requests: List[Tuple[list, torch.Tensor, Optional[torch.Tensor]]]) -> Tuple[List[torch.Tensor], Dict[str, float]]:
        """requests: list of (images, tokens, noise) request SETS (each of a prepared batch size). Stage A of set
        n+1 overlaps stage B of set n. Returns the outputs and host-side timings."""
        n = len(requests)
        hosts = [self.host_inputs(*r) for r in requests]
        bs = [h[3] for h in hosts]
        outs: List[Optional[torch.Tensor]] = [None] * n
        t_submit = [0.0] * n
        t_done = [0.0] * n
        t_start = time.perf_counter()
        # prime: A(0) [+ with recv-ahead: B receives set 0's K/V eagerly into cache set 0]
        t_submit[0] = time.perf_counter()
        self.write_prefix_inputs(bs[0], hosts[0][0], hosts[0][1])
        self.launch_prefix(bs[0])
        if self.recv_ahead:
            self._recv_caches(bs[0], 0)
        if self.recv_cq1:
            self._launch_recv(bs[0], 0)
        for i in range(n):
            s = i % self.num_sets
            b = bs[i]
            self.write_expert_inputs(b, hosts[i][2])
            if self.recv_cq1:
                # B(i) on CQ0 after its K/V landed (event from CQ1); then A(i+1) and the CQ1 recv of set i+1
                # overlap B(i)'s compute. Set (i+1)%2 is free: B(i-1) was read back before this iteration.
                self.launch_expert(b, s)
                if i + 1 < n:
                    t_submit[i + 1] = time.perf_counter()
                    self.write_prefix_inputs(bs[i + 1], hosts[i + 1][0], hosts[i + 1][1])
                    self.launch_prefix(bs[i + 1])
                    self._launch_recv(bs[i + 1], 1 - s)
            else:
                if i + 1 < n:
                    t_submit[i + 1] = time.perf_counter()
                    self.write_prefix_inputs(bs[i + 1], hosts[i + 1][0], hosts[i + 1][1])
                    self.launch_prefix(bs[i + 1])  # A(i+1) overlaps B(i); B(i)'s trace receives its own K/V first
                elif self.recv_ahead:
                    # last set: B(i)'s trace still contains the prefetch recvs -> feed them a dummy A pass
                    self.launch_prefix(b)
                self.launch_expert(b, s)  # B(i): receives set i's K/V, computes; with recv-ahead prefetches set i+1
            outs[i] = self.read_expert(b, s)  # blocks until B(i) is done
            t_done[i] = time.perf_counter()
        total = time.perf_counter() - t_start
        lat = [(t_done[i] - t_submit[i]) * 1000 for i in range(n)]
        n_req = sum(bs)
        timing = {
            "requests": n_req,
            "total_ms": total * 1000,
            "ms_per_request": total * 1000 / n_req,
            "requests_per_s": n_req / total,
            "latency_ms_median": sorted(lat)[len(lat) // 2],
            "latency_ms_first": lat[0],
        }
        return outs, timing

    def measure_stage_times(self, images, tokens, noise=None, iters: int = 6) -> Dict[str, float]:
        """Serial (A then B) time per set and the closed-pipeline steady-state time per set for the batch size of
        ``tokens``; their difference is a lower bound on the expert stage (used by the server to decide how long it
        may wait for the next request set while the expert runs)."""
        serial = []
        for _ in range(iters):
            t0 = time.perf_counter()
            self.run_one(images, tokens, noise)
            serial.append((time.perf_counter() - t0) * 1000)
        _, timing = self.run_pipelined([(images, tokens, noise)] * (iters + 1))
        sets = iters + 1
        return {
            "serial_ms": sorted(serial)[len(serial) // 2],
            "pipelined_ms": timing["total_ms"] / sets,
            "expert_ms_lower_bound": max(0.0, sorted(serial)[len(serial) // 2] - timing["total_ms"] / sets),
        }

    def close(self):
        """Ordered teardown BEFORE the parent mesh is closed: traces, sockets, buffers, models, then the two
        sub-meshes (the tests close sub-meshes explicitly before their parent). A parent close with live
        sockets / sub-meshes left an active Ethernet core wedged after a containerised server stopped
        (the next open timed out on it), while the same shutdown on the host was fine."""
        import gc

        self.release()
        for dev in (self.A, self.B):
            try:
                ttnn.synchronize_device(dev)
            except Exception:  # noqa: BLE001
                pass
        self.send_socket = None
        self.recv_socket = None
        self._in = {}
        self.cache_sets = {}
        self._out_b = {}
        self.prefix_model = None
        self.expert_model = None
        gc.collect()
        for name in ("A", "B"):
            dev = getattr(self, name)
            if dev is not None:
                try:
                    ttnn.close_mesh_device(dev)
                except Exception as e:  # noqa: BLE001
                    print(f"[pi0.5] closing sub-mesh {name}: {e}", flush=True)
                setattr(self, name, None)

    def release(self):
        for b in self.batches:
            if self._trace_a[b] is not None:
                ttnn.release_trace(self.A, self._trace_a[b])
                self._trace_a[b] = None
            for s in range(self.num_sets):
                if self._trace_b[b][s] is not None:
                    ttnn.release_trace(self.B, self._trace_b[b][s])
                    self._trace_b[b][s] = None
                if self._recv_trace[b][s] is not None:
                    ttnn.release_trace(self.B, self._recv_trace[b][s])
                    self._recv_trace[b][s] = None
