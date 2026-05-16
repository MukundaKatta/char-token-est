# char-token-est

[![crates.io](https://img.shields.io/crates/v/char-token-est.svg)](https://crates.io/crates/char-token-est)
[![docs.rs](https://img.shields.io/docsrs/char-token-est)](https://docs.rs/char-token-est)

Tokenless token-count estimator for LLM prompts. ~10% accurate on typical
prompts, fast, zero deps. Use when a real BPE tokenizer is too heavy
(routing, budget gates, log lines, progress bars).

## Usage

```rust
use char_token_est::{estimate, Family};

let n = estimate("The quick brown fox jumps over the lazy dog.", Family::Gpt);
println!("~{n} tokens");
```

Or supply your own ratio:

```rust
use char_token_est::estimate_with_ratio;
let n = estimate_with_ratio("...", 4.0);
```

## Calibration

| Family | chars/token |
| --- | --- |
| `Gpt` | 4.0 |
| `Claude` | 3.5 |
| `Gemini` | 4.0 |
| `Llama` | 3.7 |
| `Cohere` | 3.8 |

Calibration is best-effort on English + code + JSON. Pure non-Latin
input deviates further; use a real tokenizer for billing.

## License

MIT or Apache-2.0.

## Repository Health

This repository includes a dependency-free health check for core documentation, metadata, and CI wiring. Run it locally before publishing changes:

```sh
python3 scripts/check_repository_health.py
```

The same check runs in GitHub Actions on pushes and pull requests.
