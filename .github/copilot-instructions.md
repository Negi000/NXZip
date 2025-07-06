# GitHub Copilot Instructions for NXZip

<!-- Use this file to provide workspace-specific custom instructions to Copilot. For more details, visit https://code.visualstudio.com/docs/copilot/copilot-customization#_use-a-githubcopilotinstructionsmd-file -->

## Project Overview

This is the **NXZip** project - a next-generation integrated archive system that combines:
- **SPE (Structure-Preserving Encryption)** - Proprietary encryption that maintains data structure
- **High-efficiency lossless compression** - Using LZMA2/Zstd with custom optimizations
- **Enhanced security mechanisms** - Multi-layer encryption with AES-GCM/XChaCha20

## Code Style Guidelines

### Rust Conventions
- Use `snake_case` for function and variable names
- Use `PascalCase` for types and structs
- Prefer `Result<T, E>` for error handling
- Use `#[derive(Debug, Clone)]` where appropriate
- Follow Rust's ownership patterns strictly

### Architecture Patterns
- **Modular Design**: Each feature should be in its own module
- **Error Handling**: Use custom error types with `thiserror`
- **Async/Await**: Use tokio for I/O operations
- **Performance**: Optimize for memory usage and CPU efficiency

### Security Considerations
- Never log sensitive data (passwords, keys, encrypted content)
- Use secure random number generation (`rand::thread_rng()`)
- Implement constant-time comparisons for cryptographic operations
- Validate all input data before processing

### File Format Specifications
- `.nxz` files: Standard compressed format with SPE encryption
- `.nxz.sec` files: Enhanced security with additional encryption layer
- All binary data should use little-endian byte order
- Include integrity checksums (SHA256/BLAKE3) in file headers

### Testing Requirements
- Unit tests for all compression/decompression functions
- Integration tests for CLI commands
- Benchmark tests for performance validation
- Security tests for cryptographic functions

## Modules Overview

- `cli/`: Command-line interface implementation
- `engine/`: Core compression and decompression algorithms
- `crypto/`: Encryption, decryption, and key management
- `utils/`: Helper functions for hashing, metadata, etc.
- `formats/`: File format definitions and parsers

## Performance Targets

- Compression ratio: Better than LZMA2 in most cases
- Decompression speed: Optimized for SSD storage
- Memory usage: Efficient for large files (streaming where possible)
- Security: Minimal performance impact from encryption layers
