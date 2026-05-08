# ONNX Runtime Memory Subsystem

This directory contains the initial scaffolding for the ONNX Runtime
memory‑introspection and policy‑management subsystem.

## Components

### MemoryInfo
Lightweight struct describing memory domains, device types, and allocation classes.

### MemoryDiagnostics
Interfaces for EPs to publish memory usage snapshots.

### MemoryPolicyManager
Controls memory growth, fragmentation strategies, and allocation hints.

### MemoryEvents
Eventing interface for logging, tracing, and telemetry.

### MemoryRegistry
Registry for EPs to register memory domains and capabilities.

## Status
This is the initial structural scaffolding. Functional integration will be added in future iterations.
