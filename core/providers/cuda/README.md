# CUDA Provider Extensions

This folder contains additional CUDA training utilities:

- cuda_training_stream.* : Dedicated CUDA stream for training workloads
- cuda_memory_stats.* : GPU memory telemetry
- cuda_provider_debug.* : Device info and debugging helpers
- activation_workspace.* : Training activation memory workspace
- activation_suballocator.* : GPU suballocator for activation buffers

These components extend the CUDA EP with improved memory management,
debugging, and training support.
