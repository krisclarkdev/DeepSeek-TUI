use std::path::Path;

use crate::config::DEFAULT_TEXT_MODEL;
use crate::models::{
    Message,
    context_window_for_model,
};

use super::*;

/// Configuration for conversation compaction behavior.
///
/// v0.8.11 simplified this from the prior token-OR-message-count trigger
/// to a token-only trigger gated by an absolute floor. The
/// `message_threshold` field was removed: its only purpose was to fire
/// compaction on long sessions of small messages, which is exactly the
/// case where rewriting the V4 prefix cache is least valuable. Token
/// budget is the right signal; message count was a 128K-era heuristic.
#[derive(Debug, Clone, PartialEq)]
pub struct CompactionConfig {
    pub enabled: bool,
    pub token_threshold: usize,
    pub model: String,
    pub cache_summary: bool,
    /// Hard floor — `should_compact` returns `false` when total session
    /// tokens fall below this number, regardless of `enabled` or
    /// `token_threshold`. Defaults to [`MINIMUM_AUTO_COMPACTION_TOKENS`]
    /// (500K) for v0.8.11+. Tests that want to exercise the threshold
    /// logic at small fixture sizes can set this to `0` to disable the
    /// floor.
    pub auto_floor_tokens: usize,
}

impl Default for CompactionConfig {
    fn default() -> Self {
        Self {
            // ON BY DEFAULT since v0.8.6 (#402 P0 survivability) — but the
            // engine-level `auto_compact` setting was flipped OFF in v0.8.11
            // (#665) so this default is mostly a fallback for code paths
            // that build a `CompactionConfig` without going through
            // `compaction_threshold_for_model_and_effort`. Real per-model
            // values are still derived through that helper.
            enabled: true,
            // v0.8.11: 50K was a 128K-era leftover that biased every
            // unconfigured caller toward "compact almost immediately on V4."
            // Bumped to 800K (80% of V4's 1M window) so the dead-code
            // default no longer lies. Real call sites override this via
            // `compaction_threshold_for_model_and_effort`.
            token_threshold: 800_000,
            model: DEFAULT_TEXT_MODEL.to_string(),
            cache_summary: true,
            auto_floor_tokens: MINIMUM_AUTO_COMPACTION_TOKENS,
        }
    }
}

/// Hard floor for automatic compaction in v0.8.11+.
///
/// Below this token count, `should_compact` returns `false` regardless of
/// `enabled` or `token_threshold`. The point of the floor is V4 prefix-cache
/// economics: compaction rewrites the stable prefix, which destroys the KV
/// cache. At low token counts the prefix cache is healthy and compaction's
/// cost (full re-prefill at miss prices) dwarfs its benefit (a tiny budget
/// reclaim). Above the floor compaction can still be net-positive — cache
/// is already pressured, the prefix has drifted, and freeing budget matters.
///
/// Manual `/compact` slash command bypasses this floor with explicit user
/// agency.
///
/// Constant rather than configurable for v0.8.11. If anyone needs to dial
/// it (smaller models, opinionated workflows), we can add a setting later.
pub const MINIMUM_AUTO_COMPACTION_TOKENS: usize = 500_000;

pub const KEEP_RECENT_MESSAGES: usize = 4;
pub(crate) const RECENT_WORKING_SET_WINDOW: usize = 12;
pub(crate) const MAX_WORKING_SET_PATHS: usize = 24;
pub(crate) const MIN_SUMMARIZE_MESSAGES: usize = 6;
pub(crate) const SUMMARY_TEXT_SNIPPET_CHARS: usize = 800;
pub(crate) const SUMMARY_TOOL_RESULT_SNIPPET_CHARS: usize = 240;
pub(crate) const SUMMARY_INPUT_MAX_CHARS: usize = 24_000;
pub(crate) const SUMMARY_INPUT_HEAD_CHARS: usize = 14_000;
pub(crate) const SUMMARY_INPUT_TAIL_CHARS: usize = 6_000;
pub(crate) const LARGE_CONTEXT_SUMMARY_TEXT_SNIPPET_CHARS: usize = 2_000;
pub(crate) const LARGE_CONTEXT_SUMMARY_TOOL_RESULT_SNIPPET_CHARS: usize = 4_000;
pub(crate) const LARGE_CONTEXT_SUMMARY_INPUT_MAX_CHARS: usize = 120_000;
pub(crate) const LARGE_CONTEXT_SUMMARY_INPUT_HEAD_CHARS: usize = 72_000;
pub(crate) const LARGE_CONTEXT_SUMMARY_INPUT_TAIL_CHARS: usize = 36_000;
pub(crate) const LARGE_CONTEXT_SUMMARY_MAX_TOKENS: u32 = 2_048;
pub(crate) const LARGE_CONTEXT_WINDOW_TOKENS: u32 = 500_000;
pub(crate) const CACHE_ALIGNED_SUMMARY_CONTEXT_BUDGET_PERCENT: usize = 85;

#[derive(Debug, Clone, Copy)]
pub(crate) struct SummaryInputLimits {
    pub(crate) text_snippet_chars: usize,
    pub(crate) tool_result_snippet_chars: usize,
    pub(crate) input_max_chars: usize,
    pub(crate) input_head_chars: usize,
    pub(crate) input_tail_chars: usize,
    pub(crate) max_tokens: u32,
    pub(crate) word_limit: usize,
}

pub(crate) fn summary_input_limits_for_model(model: &str) -> SummaryInputLimits {
    let is_large_context =
        context_window_for_model(model).is_some_and(|window| window >= LARGE_CONTEXT_WINDOW_TOKENS);
    if is_large_context {
        SummaryInputLimits {
            text_snippet_chars: LARGE_CONTEXT_SUMMARY_TEXT_SNIPPET_CHARS,
            tool_result_snippet_chars: LARGE_CONTEXT_SUMMARY_TOOL_RESULT_SNIPPET_CHARS,
            input_max_chars: LARGE_CONTEXT_SUMMARY_INPUT_MAX_CHARS,
            input_head_chars: LARGE_CONTEXT_SUMMARY_INPUT_HEAD_CHARS,
            input_tail_chars: LARGE_CONTEXT_SUMMARY_INPUT_TAIL_CHARS,
            max_tokens: LARGE_CONTEXT_SUMMARY_MAX_TOKENS,
            word_limit: 900,
        }
    } else {
        SummaryInputLimits {
            text_snippet_chars: SUMMARY_TEXT_SNIPPET_CHARS,
            tool_result_snippet_chars: SUMMARY_TOOL_RESULT_SNIPPET_CHARS,
            input_max_chars: SUMMARY_INPUT_MAX_CHARS,
            input_head_chars: SUMMARY_INPUT_HEAD_CHARS,
            input_tail_chars: SUMMARY_INPUT_TAIL_CHARS,
            max_tokens: 1_024,
            word_limit: 500,
        }
    }
}

pub fn should_compact(
    messages: &[Message],
    config: &CompactionConfig,
    workspace: Option<&Path>,
    external_pins: Option<&[usize]>,
    external_working_set_paths: Option<&[String]>,
) -> bool {
    if !config.enabled {
        return false;
    }

    // v0.8.11: hard floor enforcement. Below the floor (default 500K tokens
    // — see `MINIMUM_AUTO_COMPACTION_TOKENS`), automatic compaction is
    // refused because rewriting the prefix kills V4's prefix cache for
    // little budget recovery. Manual `/compact` and the `compact_now` tool
    // bypass this floor by going through different code paths.
    if config.auto_floor_tokens > 0 {
        let total_session_tokens: usize = messages
            .iter()
            .map(|m| estimate_tokens_for_message(m, false))
            .sum();
        if total_session_tokens < config.auto_floor_tokens {
            return false;
        }
    }

    let plan = plan_compaction(
        messages,
        workspace,
        KEEP_RECENT_MESSAGES,
        external_pins,
        external_working_set_paths,
    );
    let pinned_tokens: usize = plan
        .pinned_indices
        .iter()
        .map(|&idx| estimate_tokens_for_message(&messages[idx], false))
        .sum();

    let token_estimate: usize = plan
        .summarize_indices
        .iter()
        .map(|&idx| estimate_tokens_for_message(&messages[idx], false))
        .sum();
    let message_count = plan.summarize_indices.len();

    // Pinned messages consume part of the budget, so compact earlier when needed.
    let effective_token_threshold = config.token_threshold.saturating_sub(pinned_tokens);

    // Token-only trigger (v0.8.11): the prior message-count branch was a
    // 128K-era heuristic that fired compaction on long chats of small
    // messages — exactly the case where rewriting the V4 prefix cache is
    // most wasteful. Token budget is the only signal that maps to actual
    // model context pressure.
    if effective_token_threshold == 0 {
        return message_count >= MIN_SUMMARIZE_MESSAGES;
    }
    if message_count < MIN_SUMMARIZE_MESSAGES {
        return false;
    }
    token_estimate > effective_token_threshold
}

