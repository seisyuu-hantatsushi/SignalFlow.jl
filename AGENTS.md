## Overview
This project is intended to develop a collection of modules for performing Software-Defined Radio (SDR) and signal processing in Julia.
## Implementation approach
- Each module performs its processing in a separate thread.
- Data exchange between threads is handled using RingBuffers.jl.
- Loss of IQ data and demodulated data must never occur.
- To guarantee this requirement, log output and GUI updates may be degraded or skipped without real-time guarantees.
## Tools
"python3" is instead of python.
