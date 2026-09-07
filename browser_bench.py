#!/usr/bin/env python3
"""
Benchmark Playwright (Chromium) in headless vs. headful mode across three sites of increasing complexity, measuring
browser startup time, page-load time, peak RAM (RSS), and CPU time. Prints a summary table and writes the raw results
to *browser_bench_results.json*.

Usage:
    python browser_bench.py

Caveats:
  - Single machine, single sitting (see `machine` block in the JSON output) -- not a controlled lab.
  - nav_s is time-to-domcontentloaded, not full page load; some sites keep loading async resources after this point.
  - cpu_s is total CPU time (user + system, summed across every process in the tree) consumed during the 2s sampling
    window after goto() returns -- not wall-clock time, so it can exceed 2s when multiple cores are busy at once.
  - rss_mb is the max RSS seen across the ~8 samples taken every 0.25s for 2s after goto() returns.
  - rss_mb sums the whole process tree (Playwright's Node driver + all Chromium subprocesses), so it's not purely
    'browser' memory -- but that overhead is identical in both modes, so the headless-vs-headful delta is still an
    apples-to-apples comparison.
"""

from __future__ import annotations

import json
import os
import platform
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from importlib.metadata import version as pkg_version
from pathlib import Path

import psutil
from playwright.sync_api import Error as PlaywrightError
from playwright.sync_api import sync_playwright

# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------

SITES = [
    ("simple", "https://example.com"),
    ("medium", "https://en.wikipedia.org/wiki/Main_Page"),
    ("complex", "https://github.com/trending"),
]
MODES = [("headless", True), ("headful", False)]

TRIALS = 10                # runs per (site, mode) -- averaged, gives a stable mean +/- stdev
NAV_TIMEOUT_MS = 15_000    # keep a slow/flaky site from stalling the run
MONITOR_DURATION_S = 2.0   # window over which RAM/CPU are sampled, starting right after goto() returns
MONITOR_INTERVAL_S = 0.25  # sampling cadence within that window (~8 samples/run)
MAX_RETRIES = 2            # extra attempts if a run fails (timeout, crash, ...)

OUT_DIR = Path(__file__).parent
RESULTS_PATH = OUT_DIR / "browser_bench_results.json"

SITE_ORDER = [s[0] for s in SITES]
MODE_ORDER = [m[0] for m in MODES]


@dataclass
class RunResult:
    site: str
    url: str
    mode: str
    trial: int
    startup_s: float
    nav_s: float
    rss_mb: float
    cpu_s: float
    ok: bool
    error: str = ""
    retries: int = 0


# --------------------------------------------------------------------------
# Measurement
# --------------------------------------------------------------------------

def _short_platform() -> str:
    if platform.system() == "Darwin":
        return f"macOS {platform.mac_ver()[0]} ({platform.machine()})"
    return platform.platform()


def get_machine_info(chromium_version: str) -> dict:
    vm = psutil.virtual_memory()
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "platform": platform.platform(),
        "platform_short": _short_platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpu_count_physical": psutil.cpu_count(logical=False),
        "cpu_count_logical": psutil.cpu_count(logical=True),
        "total_ram_gb": round(vm.total / (1024 ** 3), 1),
        "python_version": sys.version.split()[0],
        "playwright_version": pkg_version("playwright"),
        "chromium_version": chromium_version,
    }


def monitor_process_tree(root_pid: int, duration_s: float, interval_s: float) -> tuple[float, float]:
    """
    Poll RSS (MB) and CPU time (s) across this process and all its descendants (the Playwright driver + spawned
    browser/renderer processes) repeatedly over `duration_s`, returning (peak_rss_mb, cpu_time_s): the true peak RSS
    across samples, and the total user+system CPU time the whole tree consumed during the window.

    Polling (rather than one snapshot) catches child processes that spawn mid-load (e.g. a new renderer for a late
    iframe) and lets us capture a process's last-known CPU time even if it exits before the window ends.
    """
    tracked: dict[int, psutil.Process] = {}
    baseline_cpu_s: dict[int, float] = {}  # pid -> user+sys cpu time at first sighting
    last_cpu_s: dict[int, float] = {}      # pid -> most recently measured user+sys cpu time

    def live_pids() -> dict[int, psutil.Process]:
        try:
            root = psutil.Process(root_pid)
        except psutil.NoSuchProcess:
            return {}
        procs = [root] + root.children(recursive=True)
        return {p.pid: p for p in procs}

    def cpu_total_s(p: psutil.Process) -> float:
        t = p.cpu_times()
        return t.user + t.system

    def refresh() -> None:
        """
        Add newly-seen pids to `tracked` (recording their starting CPU time as a baseline) and drop exited pids from
        `tracked` -- their last-known reading stays in `last_cpu_s`/`baseline_cpu_s` so the final tally still accounts
        for CPU time they used before exiting.
        """
        current = live_pids()
        for pid, p in current.items():
            if pid not in tracked:
                try:
                    c = cpu_total_s(p)
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
                tracked[pid] = p
                baseline_cpu_s[pid] = c
                last_cpu_s[pid] = c
        for pid in list(tracked):
            if pid not in current:
                del tracked[pid]

    refresh()  # prime the initial process tree

    peak_rss_mb = 0.0
    deadline = time.perf_counter() + duration_s
    while time.perf_counter() < deadline:
        time.sleep(interval_s)
        refresh()  # picks up any process spawned mid-load
        rss_mb = 0.0
        for pid, p in list(tracked.items()):
            try:
                rss_mb += p.memory_info().rss / (1024 * 1024)
                last_cpu_s[pid] = cpu_total_s(p)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                tracked.pop(pid, None)
        peak_rss_mb = max(peak_rss_mb, rss_mb)

    cpu_time_s = sum(last_cpu_s[pid] - baseline_cpu_s[pid] for pid in last_cpu_s)
    return peak_rss_mb, cpu_time_s


def run_once(playwright, mode_name: str, headless: bool, site_name: str, url: str, trial: int) -> RunResult:
    browser = None
    try:
        t0 = time.perf_counter()
        browser = playwright.chromium.launch(headless=headless)
        startup_s = time.perf_counter() - t0

        page = browser.new_page()
        t1 = time.perf_counter()
        page.goto(url, wait_until="domcontentloaded", timeout=NAV_TIMEOUT_MS)
        nav_s = time.perf_counter() - t1

        rss_mb, cpu_s = monitor_process_tree(os.getpid(), MONITOR_DURATION_S, MONITOR_INTERVAL_S)

        return RunResult(site_name, url, mode_name, trial, startup_s, nav_s, rss_mb, cpu_s, True)
    except (PlaywrightError, TimeoutError) as e:
        return RunResult(site_name, url, mode_name, trial, -1.0, -1.0, 0.0, 0.0, False, str(e)[:200])
    finally:
        if browser is not None:
            try:
                browser.close()
            except PlaywrightError:
                pass


def run_with_retries(playwright, mode_name, headless, site_name, url, trial) -> RunResult:
    result = run_once(playwright, mode_name, headless, site_name, url, trial)
    attempts = 0
    while not result.ok and attempts < MAX_RETRIES:
        attempts += 1
        print(f"    retrying after failure: {result.error[:80]}")
        result = run_once(playwright, mode_name, headless, site_name, url, trial)
    result.retries = attempts
    return result


def run_benchmark() -> tuple[list[RunResult], dict]:
    results = []
    with sync_playwright() as p:
        probe = p.chromium.launch(headless=True)
        machine_info = get_machine_info(probe.version)
        probe.close()

        for mode_name, headless in MODES:
            for site_name, url in SITES:
                for trial in range(1, TRIALS + 1):
                    print(f"[{mode_name:8s}] {site_name:8s} trial {trial}/{TRIALS} -> {url}")
                    r = run_with_retries(p, mode_name, headless, site_name, url, trial)
                    status = "ok" if r.ok else f"FAILED ({r.error[:60]})"
                    print(
                        f"    startup={r.startup_s:.2f}s  nav={r.nav_s:.2f}s  "
                        f"rss={r.rss_mb:.0f}MB  cpu={r.cpu_s:.2f}s  [{status}]"
                    )
                    results.append(r)
    return results, machine_info


# --------------------------------------------------------------------------
# Aggregation
# --------------------------------------------------------------------------

METRICS = ("startup", "nav", "rss", "cpu")


def _mean_std(vals: list[float]) -> tuple[float, float]:
    if not vals:
        return float("nan"), float("nan")
    mean = statistics.mean(vals)
    std = statistics.stdev(vals) if len(vals) > 1 else 0.0
    return mean, std


def aggregate(results: list[RunResult]) -> dict[tuple[str, str], dict[str, dict[str, float]]]:
    """Returns {(site, mode): {metric: {"mean": ..., "std": ..., "n": ...}}}."""
    buckets: dict[tuple[str, str], dict[str, list[float]]] = {}
    counts: dict[tuple[str, str], int] = {}
    for r in results:
        key = (r.site, r.mode)
        counts[key] = counts.get(key, 0) + 1
        if not r.ok:
            continue
        b = buckets.setdefault(key, {m: [] for m in METRICS})
        b["startup"].append(r.startup_s)
        b["nav"].append(r.nav_s)
        b["rss"].append(r.rss_mb)
        b["cpu"].append(r.cpu_s)

    agg = {}
    for key, b in buckets.items():
        n_ok = len(b["startup"])
        agg[key] = {"n": n_ok, "n_total": counts[key]}
        for metric in METRICS:
            mean, std = _mean_std(b[metric])
            agg[key][metric] = {"mean": mean, "std": std}
    return agg


def print_summary(agg: dict[tuple[str, str], dict]) -> None:
    header = (
        f"{'site':8s} {'mode':9s} {'startup':>14s} {'load':>14s} "
        f"{'RAM(MB)':>16s} {'CPU(s)':>13s} {'ok':>6s}"
    )
    print("\n" + header)
    print("-" * len(header))
    for site in SITE_ORDER:
        for mode in MODE_ORDER:
            m = agg.get((site, mode))
            if not m or m["n"] == 0:
                print(f"{site:8s} {mode:9s}  no successful runs")
                continue
            s, l, r, c = m["startup"], m["nav"], m["rss"], m["cpu"]
            print(
                f"{site:8s} {mode:9s} "
                f"{s['mean']:5.2f}+/-{s['std']:4.2f}s "
                f"{l['mean']:5.2f}+/-{l['std']:4.2f}s "
                f"{r['mean']:6.0f}+/-{r['std']:5.0f} "
                f"{c['mean']:5.2f}+/-{c['std']:4.2f} "
                f"{m['n']:3d}/{m['n_total']:<3d}"
            )


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def print_reliability(results: list[RunResult]) -> None:
    total = len(results)
    failed = [r for r in results if not r.ok]
    retried = [r for r in results if r.retries > 0]
    print(
        f"\nreliability: {total - len(failed)}/{total} trials succeeded "
        f"({len(retried)} needed a retry, {len(failed)} failed even after "
        f"{MAX_RETRIES} retries)"
    )
    for r in failed:
        print(f"  {r.mode}/{r.site} trial {r.trial}: {r.error[:100]}")


def print_machine_info(machine_info: dict) -> None:
    print("machine:")
    for k, v in machine_info.items():
        print(f"  {k}: {v}")


def main() -> None:
    results, machine_info = run_benchmark()

    output = {"machine": machine_info, "runs": [asdict(r) for r in results]}
    RESULTS_PATH.write_text(json.dumps(output, indent=2))
    print(f"\nwrote {RESULTS_PATH}")

    print_machine_info(machine_info)
    print_reliability(results)

    agg = aggregate(results)
    print_summary(agg)


if __name__ == "__main__":
    main()
