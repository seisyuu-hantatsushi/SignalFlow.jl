## Overview
This project is intended to develop a collection of modules for performing Software-Defined Radio (SDR) and signal processing in Julia.
## Implementation approach
- Each module performs its processing in a separate thread.
- Data exchange between threads is handled using RingBuffers.jl.
