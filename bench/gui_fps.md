# GUI FPS Benchmark Protocol

Automated measurements run via `bench/gui_bench.py`; this document
covers the manual procedure for measuring GUI frame rate.

1. **Graph size** – load a representative graph and note the node and edge counts.
2. **Anti-aliasing** – record the anti-aliasing (AA) setting used.
3. **Labels** – specify whether labels were rendered on or off.
4. **Target FPS** – enter the target frame rate configured in the settings.
5. **Machine notes** – include CPU, GPU, operating system and driver
   versions. The automated benchmark records these automatically; manual
   runs should note them explicitly.

Record the above details alongside the observed FPS when collecting results.
