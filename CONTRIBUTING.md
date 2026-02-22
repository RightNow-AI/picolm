# Contributing to PicoLLM

Thanks for your interest in PicoLLM! This project is intentionally small (~2,500 lines of C) and we want to keep it that way.

## Ground Rules

1. **Zero dependencies.** Only libc, libm, libpthread. No external libraries.
2. **No malloc during inference.** All memory is allocated at startup.
3. **Every line must work on 256MB RAM.** If it increases memory usage, it needs a good reason.
4. **Keep it simple.** No C++, no templates, no OOP. Plain C11.

## What We Need Help With

### High Impact
- **SIMD kernels** — AVX2/AVX-512 for x86, optimized NEON for ARM
- **New quantization formats** — Q5_K fused dot product, IQ formats
- **New model architectures** — Mistral, Phi, Gemma (LLaMA-compatible)
- **Platform testing** — RISC-V boards, Pi Zero, exotic ARM SBCs

### Medium Impact
- **Grammar modes** — XML, YAML, function-call schemas (not just JSON)
- **Speculative decoding** — draft model support
- **Continuous batching** — server mode for multiple concurrent requests

### Always Welcome
- Bug fixes
- Better error messages
- Documentation improvements
- Performance measurements on new hardware

## How to Contribute

### 1. Fork & clone

```bash
git clone https://github.com/rightnow-ai/picolm.git
cd picolm/picolm
```

### 2. Build & test

```bash
make native
./picolm model.gguf -p "The capital of France is" -n 20 -t 0
# Should output: Paris. It is the largest city in France...
```

### 3. Make your changes

- One feature per PR
- Keep diffs small and focused
- Test with `make native` (x86) and ideally `make pi` (ARM) if you have one

### 4. Verify

```bash
# Build clean
make clean && make native

# Test greedy output (must match reference)
./picolm model.gguf -p "The capital of France is" -n 20 -t 0

# Test JSON mode
./picolm model.gguf --json -p "Return JSON with a name" -n 50 -t 0.3

# Check memory (should be ~45 MB for TinyLlama)
./picolm model.gguf -p "Hello" -n 10 2>&1 | grep Memory
```

### 5. Submit PR

- Clear title describing what changed
- Include test output in the PR description
- Mention which hardware you tested on

## Code Style

- C11 standard
- 4-space indentation
- `snake_case` for functions and variables
- `UPPER_CASE` for macros and constants
- `type_t` suffix for typedefs
- Comments only where the code isn't self-explanatory
- Keep functions short — if it's over 50 lines, consider splitting

## Architecture

```
picolm.c  →  model.c  →  tensor.c  →  quant.c
                ↑
          tokenizer.c    sampler.c    grammar.c
```

- **quant.c** has zero dependencies (standalone dequantization kernels)
- **tensor.c** depends on quant.c (for dequantize-in-matmul)
- **model.c** depends on tensor.c and quant.c (forward pass)
- **tokenizer.c**, **sampler.c**, **grammar.c** are independent modules
- **picolm.c** ties everything together

## Performance Tips

If you're adding SIMD code:

```c
#ifdef PICOLM_NEON
    // ARM NEON path (Pi 3/4/5)
    float32x4_t v = vld1q_f32(ptr);
    ...
#elif defined(PICOLM_SSE2)
    // x86 SSE2 path (Intel/AMD)
    __m128 v = _mm_loadu_ps(ptr);
    ...
#endif
    // Scalar fallback (always works)
    for (int i = 0; i < n; i++) { ... }
```

Always keep the scalar fallback. Never break builds on unsupported platforms.
