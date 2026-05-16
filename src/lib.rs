//! # char-token-est
//!
//! Estimate token counts from raw text without invoking a BPE tokenizer.
//!
//! Real tokenization is fast but pulls in tens of MB of vocab data. For
//! routing, budget gating, log lines, and progress bars you can get
//! within ~10% accuracy with a per-model-family chars-per-token constant
//! and no dependencies.
//!
//! ## Example
//!
//! ```
//! use char_token_est::{estimate, Family};
//! let text = "The quick brown fox jumps over the lazy dog.";
//! let n = estimate(text, Family::Gpt);
//! assert!(n >= 9 && n <= 14, "got {n}");
//! ```
//!
//! ## Calibration
//!
//! Constants are derived from average chars-per-token over a multilingual
//! corpus of typical prompts (English + code + JSON). Pure-code or
//! non-Latin inputs deviate further; pass [`estimate_with_ratio`] to
//! supply your own ratio.
//!
//! | Family | chars/token |
//! | --- | --- |
//! | `Gpt` (GPT-4/5, o3/o4 cl100k\_base) | 4.0 |
//! | `Claude` | 3.5 |
//! | `Gemini` | 4.0 |
//! | `Llama` (Llama 3 tiktoken-32k) | 3.7 |
//! | `Cohere` | 3.8 |

#![deny(missing_docs)]

/// Model family used to pick a chars-per-token ratio.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Family {
    /// GPT-4 / GPT-5 / o3 / o4 (cl100k_base, o200k_base).
    Gpt,
    /// Anthropic Claude.
    Claude,
    /// Google Gemini.
    Gemini,
    /// Meta Llama 3.
    Llama,
    /// Cohere Command R / R+.
    Cohere,
}

impl Family {
    /// Characters-per-token ratio for this family.
    pub fn chars_per_token(self) -> f64 {
        match self {
            Family::Gpt => 4.0,
            Family::Claude => 3.5,
            Family::Gemini => 4.0,
            Family::Llama => 3.7,
            Family::Cohere => 3.8,
        }
    }

    /// Best-effort guess from a model id string. Falls back to `Gpt`
    /// when nothing matches.
    pub fn guess_from_model_id(id: &str) -> Self {
        let s = id.to_ascii_lowercase();
        if s.contains("claude") {
            Family::Claude
        } else if s.contains("gemini") {
            Family::Gemini
        } else if s.contains("llama") {
            Family::Llama
        } else if s.contains("cohere") || s.contains("command-r") {
            Family::Cohere
        } else {
            Family::Gpt
        }
    }
}

/// Estimate token count for `text` using the family's chars-per-token.
pub fn estimate(text: &str, family: Family) -> u64 {
    estimate_with_ratio(text, family.chars_per_token())
}

/// Estimate token count using a caller-supplied chars-per-token ratio.
///
/// Returns at least 1 if `text` is non-empty.
pub fn estimate_with_ratio(text: &str, chars_per_token: f64) -> u64 {
    if text.is_empty() {
        return 0;
    }
    let chars = text.chars().count() as f64;
    let est = (chars / chars_per_token).ceil() as u64;
    est.max(1)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_string_is_zero() {
        assert_eq!(estimate("", Family::Gpt), 0);
    }

    #[test]
    fn ratio_picks_floor_of_one() {
        // 1 char / 4 chars-per-token -> 0.25 -> ceil = 1
        assert_eq!(estimate("a", Family::Gpt), 1);
    }

    #[test]
    fn family_guess_works() {
        assert_eq!(
            Family::guess_from_model_id("claude-sonnet-4-5"),
            Family::Claude
        );
        assert_eq!(
            Family::guess_from_model_id("meta.llama3-70b"),
            Family::Llama
        );
        assert_eq!(Family::guess_from_model_id("gemini-2.5-pro"), Family::Gemini);
        assert_eq!(
            Family::guess_from_model_id("cohere.command-r-plus"),
            Family::Cohere
        );
        assert_eq!(Family::guess_from_model_id("gpt-5"), Family::Gpt);
        assert_eq!(
            Family::guess_from_model_id("something-else"),
            Family::Gpt
        );
    }
}
