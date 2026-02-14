## Overview
- This project is intended to develop a collection of modules for performing Software-Defined Radio (SDR) and signal processing in Julia.

## Directions
-Each time a task is completed or interrupted, describe the detailed work performed in HANDOVER.md in Markdown format.
- Be sure to include the following items:
   1. Current progress status
   1. Files changed and the reasons for the changes
   1. Tasks to be addressed next
   1. Known issues

## Requirements
- Applications that use Blocks should be able to terminate gracefully upon receiving a SIGINT signal.

## Implementation approach
- FrameSync, SP phase, and TMCC phase are evaluated in stream time.
- Each module performs its processing in a separate thread.
- Data exchange between threads is handled using RingBuffers.jl.
- Loss of IQ data and demodulated data must never occur.
- To guarantee this requirement, log output and GUI updates may be degraded or skipped without real-time guarantees.
- Each block attaches a sequence number to its output frames, and the sink side verifies the consistency and continuity of the sequence numbers.

## Tools
"python3" is instead of python.
