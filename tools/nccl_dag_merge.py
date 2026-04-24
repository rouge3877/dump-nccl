#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
#
# nccl_dag_merge.py – Offline merge and analysis of NCCL DAG trace files.
#
# Usage:
#   python3 nccl_dag_merge.py nccl_dag_*.jsonl -o merged.json
#   python3 nccl_dag_merge.py nccl_dag_*.jsonl --chrome-trace trace.json
#   python3 nccl_dag_merge.py nccl_dag_*.jsonl --dot graph.dot
#   python3 nccl_dag_merge.py nccl_dag_*.jsonl --critical-path
#   python3 nccl_dag_merge.py nccl_dag_*.jsonl --summary

import argparse
import json
import sys
import os
from collections import defaultdict


def load_traces(file_paths):
    """Load DAG nodes from one or more JSONL files.
    Each file is assumed to come from a different rank (inferred from filename).
    """
    all_nodes = []
    for rank, path in enumerate(sorted(file_paths)):
        with open(path, "r") as f:
            for lineno, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    node = json.loads(line)
                except json.JSONDecodeError as e:
                    print(f"WARNING: {path}:{lineno}: bad JSON: {e}", file=sys.stderr)
                    continue
                node["rank"] = rank
                node["sourceFile"] = os.path.basename(path)
                all_nodes.append(node)
    # Sort globally by timestamp
    all_nodes.sort(key=lambda n: n.get("ts", 0))
    return all_nodes


def build_cross_rank_edges(nodes):
    """Build cross-rank edges by matching send→recv on (opCount, ch, peer).

    A send event on rank A (peer=B) matches a recv event on rank B (peer=A)
    with the same opCount and channel.
    """
    # Index send xmit events: key = (opCount, ch, senderRank)
    sends = defaultdict(list)
    recvs = defaultdict(list)

    for n in nodes:
        ev = n.get("event", "")
        if ev == "proxy_send_xmit":
            key = (n["opCount"], n["ch"], n["rank"])
            sends[key].append(n)
        elif ev == "proxy_recv_recv":
            key = (n["opCount"], n["ch"], n["rank"])
            recvs[key].append(n)

    edges = []
    for n in nodes:
        if n.get("event") != "proxy_recv_recv":
            continue
        # This recv on rank R from peer P should match a send on rank P to peer R
        recv_rank = n["rank"]
        send_rank = n.get("peer", -1)
        if send_rank < 0:
            continue
        key = (n["opCount"], n["ch"], send_rank)
        candidates = sends.get(key, [])
        if candidates:
            # Match the earliest unmatched send
            send_node = candidates.pop(0)
            edges.append(
                {
                    "from_rank": send_rank,
                    "from_id": send_node["id"],
                    "to_rank": recv_rank,
                    "to_id": n["id"],
                    "opCount": n["opCount"],
                    "ch": n["ch"],
                    "type": "cross_rank",
                }
            )
    return edges


def write_merged_json(nodes, edges, output_path):
    """Write the full merged DAG as a single JSON file."""
    merged = {
        "nodes": nodes,
        "crossRankEdges": edges,
        "metadata": {
            "numRanks": len(set(n["rank"] for n in nodes)),
            "numNodes": len(nodes),
            "numCrossEdges": len(edges),
        },
    }
    with open(output_path, "w") as f:
        json.dump(merged, f, indent=2)
    print(
        f"Merged DAG written to {output_path} "
        f"({len(nodes)} nodes, {len(edges)} cross-rank edges)"
    )


def write_chrome_trace(nodes, edges, output_path):
    """Export as Chrome Trace Event Format (chrome://tracing).

    Each layer gets its own track; ranks are shown as separate processes.
    """
    trace_events = []
    # Use the minimum timestamp as the base
    if not nodes:
        return
    ts_base = min(n["ts"] for n in nodes)

    for n in nodes:
        # Instant event per node
        ev = {
            "name": n.get("event", "unknown"),
            "cat": n.get("layer", "unknown"),
            "ph": "i",  # instant event
            "ts": (n["ts"] - ts_base) / 1000.0,  # ns → µs
            "pid": n.get("rank", 0),
            "tid": _layer_tid(n.get("layer", "")),
            "s": "t",  # scope = thread
            "args": {
                "id": n.get("id"),
                "opCount": n.get("opCount"),
                "bytes": n.get("bytes"),
                "ch": n.get("ch"),
                "peer": n.get("peer"),
                "parent": n.get("parent"),
                "detail": n.get("detail", ""),
            },
        }
        trace_events.append(ev)

    # Add cross-rank flow events
    for i, edge in enumerate(edges):
        flow_id = f"xr_{i}"
        # start (send side)
        trace_events.append(
            {
                "name": "cross_rank",
                "cat": "cross_rank",
                "ph": "s",
                "id": flow_id,
                "ts": _find_ts(nodes, edge["from_rank"], edge["from_id"], ts_base),
                "pid": edge["from_rank"],
                "tid": _layer_tid("proxy"),
            }
        )
        # end (recv side)
        trace_events.append(
            {
                "name": "cross_rank",
                "cat": "cross_rank",
                "ph": "f",
                "id": flow_id,
                "bp": "e",
                "ts": _find_ts(nodes, edge["to_rank"], edge["to_id"], ts_base),
                "pid": edge["to_rank"],
                "tid": _layer_tid("proxy"),
            }
        )

    # Process / thread name metadata
    ranks = sorted(set(n.get("rank", 0) for n in nodes))
    for r in ranks:
        trace_events.append(
            {
                "name": "process_name",
                "ph": "M",
                "pid": r,
                "args": {"name": f"Rank {r}"},
            }
        )
        for layer, tid in [("api", 0), ("kernel", 1), ("proxy", 2), ("network", 3)]:
            trace_events.append(
                {
                    "name": "thread_name",
                    "ph": "M",
                    "pid": r,
                    "tid": tid,
                    "args": {"name": layer},
                }
            )

    with open(output_path, "w") as f:
        json.dump({"traceEvents": trace_events}, f)
    print(f"Chrome trace written to {output_path}")


def _layer_tid(layer):
    return {"api": 0, "kernel": 1, "proxy": 2, "network": 3}.get(layer, 4)


def _find_ts(nodes, rank, node_id, ts_base):
    for n in nodes:
        if n.get("rank") == rank and n.get("id") == node_id:
            return (n["ts"] - ts_base) / 1000.0
    return 0


def write_dot(nodes, edges, output_path):
    """Export as DOT graph for Graphviz visualization."""
    with open(output_path, "w") as f:
        f.write("digraph nccl_dag {\n")
        f.write("  rankdir=TB;\n")
        f.write("  node [shape=box, fontsize=10];\n\n")

        # Subgraph per rank
        ranks = sorted(set(n.get("rank", 0) for n in nodes))
        for r in ranks:
            f.write(f"  subgraph cluster_rank{r} {{\n")
            f.write(f'    label="Rank {r}";\n')
            rank_nodes = [n for n in nodes if n.get("rank") == r]
            for n in rank_nodes:
                nid = f"r{r}_n{n['id']}"
                label = f"{n.get('event', '?')}\\n{n.get('detail', '')}"
                color = {
                    "api": "lightblue",
                    "kernel": "lightyellow",
                    "proxy": "lightgreen",
                    "network": "lightsalmon",
                }.get(n.get("layer", ""), "white")
                f.write(
                    f'    {nid} [label="{label}", style=filled, fillcolor={color}];\n'
                )
            f.write("  }\n\n")

        # Intra-rank parent edges
        for n in nodes:
            parent = n.get("parent")
            if parent is not None and parent != (2**64 - 1):
                r = n.get("rank", 0)
                f.write(f"  r{r}_n{parent} -> r{r}_n{n['id']};\n")

        # Dependency edges
        for n in nodes:
            r = n.get("rank", 0)
            for dep in n.get("deps", []):
                if dep != (2**64 - 1):
                    f.write(f"  r{r}_n{dep} -> r{r}_n{n['id']} [style=dashed];\n")

        # Cross-rank edges
        for edge in edges:
            f.write(
                f"  r{edge['from_rank']}_n{edge['from_id']} -> "
                f"r{edge['to_rank']}_n{edge['to_id']} "
                f"[color=red, style=bold, constraint=false];\n"
            )

        f.write("}\n")
    print(f"DOT graph written to {output_path}")


def print_summary(nodes, edges):
    """Print a human-readable summary of the DAG."""
    ranks = sorted(set(n.get("rank", 0) for n in nodes))
    layers = defaultdict(int)
    events = defaultdict(int)
    total_bytes = 0

    for n in nodes:
        layers[n.get("layer", "unknown")] += 1
        events[n.get("event", "unknown")] += 1
        total_bytes += n.get("bytes", 0)

    print("=" * 60)
    print("NCCL DAG Trace Summary")
    print("=" * 60)
    print(f"  Ranks:            {len(ranks)}")
    print(f"  Total nodes:      {len(nodes)}")
    print(f"  Cross-rank edges: {len(edges)}")
    print(f"  Total bytes:      {total_bytes:,}")
    print()
    print("  Nodes by layer:")
    for layer in ["api", "kernel", "proxy", "network"]:
        if layer in layers:
            print(f"    {layer:12s}: {layers[layer]}")
    print()
    print("  Nodes by event type:")
    for ev, count in sorted(events.items(), key=lambda x: -x[1]):
        print(f"    {ev:24s}: {count}")

    if nodes:
        ts_min = min(n["ts"] for n in nodes)
        ts_max = max(n["ts"] for n in nodes)
        duration_ms = (ts_max - ts_min) / 1e6
        print(f"\n  Time span: {duration_ms:.3f} ms")
    print("=" * 60)


def find_critical_path(nodes, edges):
    """Find the longest (critical) path in the DAG by timestamp span.

    Uses a simple topological-order DP on the parent edges.
    """
    # Build adjacency: child → parent(s)
    by_rank_id = {}
    for n in nodes:
        by_rank_id[(n.get("rank", 0), n["id"])] = n

    # Build graph: node_key → list of predecessor node_keys
    predecessors = defaultdict(list)
    for n in nodes:
        r = n.get("rank", 0)
        nk = (r, n["id"])
        parent = n.get("parent")
        if parent is not None and parent != (2**64 - 1):
            pk = (r, parent)
            if pk in by_rank_id:
                predecessors[nk].append(pk)
        for dep in n.get("deps", []):
            if dep != (2**64 - 1):
                dk = (r, dep)
                if dk in by_rank_id:
                    predecessors[nk].append(dk)

    for edge in edges:
        fk = (edge["from_rank"], edge["from_id"])
        tk = (edge["to_rank"], edge["to_id"])
        if fk in by_rank_id and tk in by_rank_id:
            predecessors[tk].append(fk)

    # DP: longest path to each node
    dist = {}
    prev = {}

    def dp(nk):
        if nk in dist:
            return dist[nk]
        node = by_rank_id[nk]
        best = 0
        best_prev = None
        for pk in predecessors.get(nk, []):
            d = dp(pk)
            duration = node["ts"] - by_rank_id[pk]["ts"]
            if d + duration > best:
                best = d + duration
                best_prev = pk
        dist[nk] = best
        prev[nk] = best_prev
        return best

    # Run DP for all nodes
    for nk in by_rank_id:
        dp(nk)

    if not dist:
        print("No nodes to analyze.")
        return

    # Find the endpoint with longest path
    end_nk = max(dist, key=dist.get)
    path = []
    nk = end_nk
    while nk is not None:
        path.append(nk)
        nk = prev.get(nk)
    path.reverse()

    print("\nCritical Path (longest causal chain):")
    print("-" * 70)
    ts_base = by_rank_id[path[0]]["ts"] if path else 0
    for nk in path:
        n = by_rank_id[nk]
        rel_us = (n["ts"] - ts_base) / 1000.0
        print(
            f"  [{n.get('rank', 0)}] +{rel_us:10.1f} µs  "
            f"{n.get('layer', '?'):8s} {n.get('event', '?'):24s} "
            f"{n.get('detail', '')}"
        )
    total_us = (by_rank_id[end_nk]["ts"] - ts_base) / 1000.0
    print(f"\n  Total critical path duration: {total_us:.1f} µs")


def main():
    parser = argparse.ArgumentParser(
        description="Merge and analyze NCCL DAG trace files"
    )
    parser.add_argument("files", nargs="+", help="Input .jsonl trace files")
    parser.add_argument("-o", "--output", help="Write merged JSON to this file")
    parser.add_argument(
        "--chrome-trace", help="Export Chrome trace format to this file"
    )
    parser.add_argument("--dot", help="Export DOT graph to this file")
    parser.add_argument(
        "--summary", action="store_true", help="Print summary statistics"
    )
    parser.add_argument(
        "--critical-path", action="store_true", help="Find and print the critical path"
    )
    args = parser.parse_args()

    nodes = load_traces(args.files)
    edges = build_cross_rank_edges(nodes)

    if args.output:
        write_merged_json(nodes, edges, args.output)

    if args.chrome_trace:
        write_chrome_trace(nodes, edges, args.chrome_trace)

    if args.dot:
        write_dot(nodes, edges, args.dot)

    if args.summary:
        print_summary(nodes, edges)

    if args.critical_path:
        find_critical_path(nodes, edges)

    if not any(
        [args.output, args.chrome_trace, args.dot, args.summary, args.critical_path]
    ):
        print_summary(nodes, edges)


if __name__ == "__main__":
    main()
