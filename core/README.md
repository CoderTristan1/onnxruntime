# Core Runtime Extensions

This directory contains extensions to the ONNX Runtime core engine:

- execution_trace.* : Kernel-level execution tracing
- memory_planner_ext.* : Extension point for activation memory planners
- core_telemetry.* : Centralized runtime logging
- training_state_snapshot.* : Training state reporting

These components improve observability, debuggability, and extensibility
of the core runtime, especially for training workloads.
