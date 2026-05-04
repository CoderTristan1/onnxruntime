# Provider Infrastructure Extensions

This folder contains provider-agnostic infrastructure used by execution providers:

- provider_memory_manager.* : Unified memory manager interface
- provider_capabilities.* : Capability descriptor for EP feature sets
- provider_debug.* : Standardized debugging interface
- provider_registry.* : EP factory registration and lookup

These components improve modularity and observability across all execution providers.
