# Headless vs Headful Browser Benchmark

The benchmark launches Chromium 151 via Playwright 1.62 in both headless and headful mode and loads three sites of increasing complexity. Each site/mode combination is run 10 times, measuring browser startup time and sampling RAM (RSS) and CPU time across the entire process tree. Headless mode uses the new headless implementation, not the `chrome-headless-shell`.

All runs were on a Mac with an 18-core Apple Silicon chip and 48GB of RAM, running macOS 26.6.2.

## Try it?

1. Create a virtual environment.
2. Install the packages listed in *requirements.txt*.
3. Run the following command:
   ```sh
   $ python browser_bench.py
   ```
