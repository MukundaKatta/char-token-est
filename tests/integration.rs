use char_token_est::{estimate, estimate_with_ratio, Family};

#[test]
fn realistic_short_english_is_close_to_word_count() {
    // 9 words, 44 chars. With 4 chars/token, expect ~11.
    let text = "The quick brown fox jumps over the lazy dog.";
    let n = estimate(text, Family::Gpt);
    assert!((10..=13).contains(&n), "got {n} for: {text}");
}

#[test]
fn claude_is_higher_than_gpt_for_same_text() {
    let text = "The quick brown fox jumps over the lazy dog.";
    let gpt = estimate(text, Family::Gpt);
    let claude = estimate(text, Family::Claude);
    // Claude has lower chars-per-token -> higher token count.
    assert!(claude >= gpt, "claude {claude} should be >= gpt {gpt}");
}

#[test]
fn custom_ratio_overrides() {
    // 100 chars at 5 chars-per-token = 20 tokens.
    let text = "a".repeat(100);
    assert_eq!(estimate_with_ratio(&text, 5.0), 20);
}

#[test]
fn unicode_chars_counted_not_bytes() {
    // Each emoji is one Unicode char (multiple bytes in UTF-8).
    let text = "\u{1F600}\u{1F600}\u{1F600}\u{1F600}";
    // 4 chars / 4 chars-per-token = 1
    assert_eq!(estimate(text, Family::Gpt), 1);
}

#[test]
fn family_guess_table() {
    assert_eq!(
        Family::guess_from_model_id("anthropic.claude-sonnet-4-5"),
        Family::Claude
    );
    assert_eq!(Family::guess_from_model_id("o3-mini"), Family::Gpt);
}
