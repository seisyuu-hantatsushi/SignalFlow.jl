#!/usr/bin/env python3
import argparse
import math
import re
from pathlib import Path
from collections import defaultdict


def mean_std(vals):
    if not vals:
        return float('nan'), float('nan')
    m = sum(vals) / len(vals)
    v = sum((x - m) ** 2 for x in vals) / len(vals)
    return m, math.sqrt(v)


def parse_log(path: Path):
    text = path.read_text(errors="ignore")
    evm_vals = [float(x) for x in re.findall(r"EVM\[DataCarriers:qpsk\]: evm=([0-9.]+)%", text)]
    h_vals = [float(x) for x in re.findall(r"mean\|H\|=([0-9.]+)", text)]
    lock = len(re.findall(r"ISDBTFrameSync: lock corr=", text))
    unlock = len(re.findall(r"ISDBTFrameSync: unlock corr=", text))
    forced = len(re.findall(r"forced_resync", text))
    outlier = len(re.findall(r"outlier_resync", text))
    phase_updated = len(re.findall(r"PhaseSlope: .*updated=true", text))
    cpe_updated = len(re.findall(r"CPE: .*updated=true", text))
    conf_vals = [float(x) for x in re.findall(r"CPE: .*?conf=([0-9.]+)", text)]
    def last_u64(pattern: str):
        m = re.findall(pattern, text)
        return int(m[-1]) if m else 0

    phase_skip_gate = last_u64(r"PhaseSlope: .*skip_gate=([0-9]+)")
    phase_skip_fit_input = last_u64(r"PhaseSlope: .*skip_fit_input=([0-9]+)")
    phase_skip_fit_rms = last_u64(r"PhaseSlope: .*skip_fit_rms=([0-9]+)")
    phase_skip_small = last_u64(r"PhaseSlope: .*skip_small_delta=([0-9]+)")
    phase_skip_invalid = last_u64(r"PhaseSlope: .*skip_invalid_fit=([0-9]+)")
    cpe_skip_gate = last_u64(r"CPE: .*skip_gate=([0-9]+)")
    cpe_skip_no_used = last_u64(r"CPE: .*skip_no_used=([0-9]+)")
    cpe_skip_small_err = last_u64(r"CPE: .*skip_small_err=([0-9]+)")
    cpe_skip_zero_delta = last_u64(r"CPE: .*skip_zero_delta=([0-9]+)")
    sink_fail = None
    m = re.findall(r"FFTBlock input stats: .*sink_fail=([0-9]+)", text)
    if m:
        sink_fail = int(m[-1])
    shutdown = "Shutdown complete." in text

    alpha = None
    snr = None
    m_alpha = re.search(r"alpha0p([0-9]+)", path.name)
    if m_alpha:
        alpha = float(f"0.{m_alpha.group(1)}")
    m_snr = re.search(r"awgn(-?[0-9]+)", path.name)
    if m_snr:
        snr = int(m_snr.group(1))

    evm_mean, evm_std = mean_std(evm_vals)
    h_mean, h_std = mean_std(h_vals)
    conf_mean, conf_std = mean_std(conf_vals)

    return {
        "file": path.name,
        "alpha": alpha,
        "snr": snr,
        "evm_n": len(evm_vals),
        "evm_mean": evm_mean,
        "evm_std": evm_std,
        "h_n": len(h_vals),
        "h_mean": h_mean,
        "h_std": h_std,
        "lock": lock,
        "unlock": unlock,
        "forced": forced,
        "outlier": outlier,
        "phase_updated": phase_updated,
        "cpe_updated": cpe_updated,
        "phase_skip_gate": phase_skip_gate,
        "phase_skip_fit_input": phase_skip_fit_input,
        "phase_skip_fit_rms": phase_skip_fit_rms,
        "phase_skip_small": phase_skip_small,
        "phase_skip_invalid": phase_skip_invalid,
        "cpe_skip_gate": cpe_skip_gate,
        "cpe_skip_no_used": cpe_skip_no_used,
        "cpe_skip_small_err": cpe_skip_small_err,
        "cpe_skip_zero_delta": cpe_skip_zero_delta,
        "conf_mean": conf_mean,
        "conf_std": conf_std,
        "sink_fail": sink_fail,
        "shutdown": shutdown,
    }


def fmt(x, nd=2):
    if x != x:
        return "-"
    return f"{x:.{nd}f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("logs", nargs="+")
    args = ap.parse_args()

    rows = [parse_log(Path(p)) for p in args.logs]
    rows.sort(key=lambda r: (r["snr"], r["alpha"], r["file"]))

    print("file | snr | alpha | evm(n/mean/std) | H(n/mean/std) | lock/unlock | forced/outlier | phase_up/cpe_up | phase_skip(g/in/s/rms/inv) | cpe_skip(g/n/s/z) | conf_mean | sink_fail | shutdown")
    print("-" * 170)
    for r in rows:
        print(
            f"{r['file']} | {r['snr']} | {r['alpha']} | "
            f"{r['evm_n']}/{fmt(r['evm_mean'])}/{fmt(r['evm_std'])} | "
            f"{r['h_n']}/{fmt(r['h_mean'],4)}/{fmt(r['h_std'],4)} | "
            f"{r['lock']}/{r['unlock']} | {r['forced']}/{r['outlier']} | "
            f"{r['phase_updated']}/{r['cpe_updated']} | "
            f"{r['phase_skip_gate']}/{r['phase_skip_fit_input']}/{r['phase_skip_small']}/{r['phase_skip_fit_rms']}/{r['phase_skip_invalid']} | "
            f"{r['cpe_skip_gate']}/{r['cpe_skip_no_used']}/{r['cpe_skip_small_err']}/{r['cpe_skip_zero_delta']} | "
            f"{fmt(r['conf_mean'],3)} | {r['sink_fail'] if r['sink_fail'] is not None else '-'} | {int(r['shutdown'])}"
        )

    by_alpha = defaultdict(list)
    for r in rows:
        if r["alpha"] is not None:
            by_alpha[r["alpha"]].append(r)

    print("\nalpha summary (lower evm_mean is better)")
    print("alpha | runs | evm_mean(avg) | evm_mean(std_across_runs) | unlock_sum | sink_fail_sum")
    print("-" * 90)
    for alpha in sorted(by_alpha.keys()):
        rs = by_alpha[alpha]
        evm_means = [r["evm_mean"] for r in rs if r["evm_n"] > 0]
        m, s = mean_std(evm_means)
        unlock_sum = sum(r["unlock"] for r in rs)
        sink_sum = sum((r["sink_fail"] or 0) for r in rs)
        print(f"{alpha:.1f} | {len(rs)} | {fmt(m)} | {fmt(s)} | {unlock_sum} | {sink_sum}")


if __name__ == "__main__":
    main()
