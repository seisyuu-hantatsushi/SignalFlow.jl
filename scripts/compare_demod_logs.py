#!/usr/bin/env python3
import argparse
import re
import statistics
from pathlib import Path


EXP_FRAME_MS = 231.336


def parse_log(path: Path):
    lines = path.read_text(errors="ignore").splitlines()
    out_idx = []
    gap_idx = []
    cycles = []
    outliers = []
    gaps = []
    recv_rates = []
    tmcc = {"norm": {"metric": [], "conf": []}, "flip": {"metric": [], "conf": []}}

    for i, line in enumerate(lines):
        def safe_float(token: str):
            try:
                return float(token)
            except ValueError:
                return None

        if "ISDBTFrameSync: lock " in line:
            pass
        m = re.search(r"recv rate: ([0-9.e+-]+)S/sec", line)
        if m:
            recv_rates.append(float(m.group(1)))
        m = re.search(r"frame_cycle_ms=([0-9]+(?:\\.[0-9]+)?)", line)
        if m:
            v = safe_float(m.group(1))
            if v is not None:
                cycles.append(v)
        m = re.search(r"frame_cycle_outlier_ms=([0-9]+(?:\\.[0-9]+)?)", line)
        if m:
            v = safe_float(m.group(1))
            if v is not None:
                outliers.append(v)
                out_idx.append(i)
        m = re.search(r"input_gap_ms=([0-9]+(?:\\.[0-9]+)?)", line)
        if m:
            v = safe_float(m.group(1))
            if v is not None:
                gaps.append(v)
                gap_idx.append(i)
        for tag in ("norm", "flip"):
            m = re.search(rf"TMCCDBPSK\[{tag}\]: bit=[01] metric=([+-]?[0-9.]+) conf=([0-9.]+)", line)
            if m:
                tmcc[tag]["metric"].append(float(m.group(1)))
                tmcc[tag]["conf"].append(float(m.group(2)))

    out_near_gap = 0
    if out_idx and gap_idx:
        out_near_gap = sum(1 for oi in out_idx if any(abs(gi - oi) <= 20 for gi in gap_idx))

    return {
        "file": path.name,
        "lock": sum("ISDBTFrameSync: lock " in l for l in lines),
        "unlock": sum("ISDBTFrameSync: unlock " in l for l in lines),
        "forced": sum("forced_resync" in l for l in lines),
        "src_bp": sum("ADFMCOMMS2Src: recv_backpressure" in l for l in lines),
        "fs_bp": sum("ISDBTFrameSync: input_backpressure" in l for l in lines),
        "cpe_bp": sum("CPE: input_backpressure" in l for l in lines),
        "slope_bp": sum("PhaseSlope: input_backpressure" in l for l in lines),
        "cycles_n": len(cycles),
        "cycles_mean": statistics.fmean(cycles) if cycles else float("nan"),
        "cycles_std": statistics.pstdev(cycles) if len(cycles) > 1 else 0.0,
        "o10": sum(abs(x - EXP_FRAME_MS) > 10.0 for x in cycles),
        "o20": sum(abs(x - EXP_FRAME_MS) > 20.0 for x in cycles),
        "out_n": len(outliers),
        "out_max": max(outliers) if outliers else float("nan"),
        "gaps_n": len(gaps),
        "gaps_max": max(gaps) if gaps else float("nan"),
        "out_near_gap": out_near_gap,
        "recv_n": len(recv_rates),
        "recv_mean": statistics.fmean(recv_rates) if recv_rates else float("nan"),
        "recv_std": statistics.pstdev(recv_rates) if len(recv_rates) > 1 else 0.0,
        "tmcc_norm_absmean": statistics.fmean(abs(x) for x in tmcc["norm"]["metric"]) if tmcc["norm"]["metric"] else float("nan"),
        "tmcc_norm_conf": statistics.fmean(tmcc["norm"]["conf"]) if tmcc["norm"]["conf"] else float("nan"),
        "tmcc_flip_absmean": statistics.fmean(abs(x) for x in tmcc["flip"]["metric"]) if tmcc["flip"]["metric"] else float("nan"),
        "tmcc_flip_conf": statistics.fmean(tmcc["flip"]["conf"]) if tmcc["flip"]["conf"] else float("nan"),
    }


def fmt(v, prec=3):
    if isinstance(v, float):
        if v != v:
            return "-"
        return f"{v:.{prec}f}"
    return str(v)


def main():
    ap = argparse.ArgumentParser(description="Compare ISDB-T demod log metrics.")
    ap.add_argument("logs", nargs="+", help="log file paths")
    args = ap.parse_args()

    metrics = []
    for p in args.logs:
        path = Path(p)
        if not path.exists():
            raise SystemExit(f"missing log: {p}")
        metrics.append(parse_log(path))

    headers = [
        "file", "cycles_n", "cy_mean", "cy_std", "o20", "out_n", "out_max",
        "gaps_max", "out~gap", "bp(src/fs/cpe/slope)", "recv_mean", "recv_std",
        "tmcc_n(abs/conf)", "tmcc_f(abs/conf)"
    ]
    print(" | ".join(headers))
    print("-" * 160)
    for m in metrics:
        row = [
            m["file"],
            str(m["cycles_n"]),
            fmt(m["cycles_mean"]),
            fmt(m["cycles_std"]),
            str(m["o20"]),
            str(m["out_n"]),
            fmt(m["out_max"]),
            fmt(m["gaps_max"]),
            f'{m["out_near_gap"]}/{m["out_n"]}',
            f'{m["src_bp"]}/{m["fs_bp"]}/{m["cpe_bp"]}/{m["slope_bp"]}',
            fmt(m["recv_mean"], 1),
            fmt(m["recv_std"], 1),
            f'{fmt(m["tmcc_norm_absmean"])} / {fmt(m["tmcc_norm_conf"])}',
            f'{fmt(m["tmcc_flip_absmean"])} / {fmt(m["tmcc_flip_conf"])}',
        ]
        print(" | ".join(row))


if __name__ == "__main__":
    main()
