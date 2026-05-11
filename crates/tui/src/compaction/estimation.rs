use anyhow::Result;
use regex::Regex;
use std::collections::{BTreeSet, HashMap, HashSet};
use std::fmt::Write;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;
use std::time::Duration;

use crate::client::DeepSeekClient;
use crate::config::DEFAULT_TEXT_MODEL;
use crate::llm_client::LlmClient;
use crate::logging;
use crate::models::{
    CacheControl, ContentBlock, Message, MessageRequest, SystemBlock, SystemPrompt,
    context_window_for_model,
};

use super::*;
use crate::compaction::*;

pub(crate) fn estimate_tokens_for_message(message: &Message, include_thinking: bool) -> usize {
    message
        .content
        .iter()
        .map(|c| match c {
            ContentBlock::Text { text, .. } => text.len() / 4,
            // Historical reasoning blocks are UI/session metadata for DeepSeek.
            // Only current-turn tool-call reasoning is sent back to the API.
            ContentBlock::Thinking { thinking } if include_thinking => thinking.len() / 4,
            ContentBlock::Thinking { .. } => 0,
            ContentBlock::ToolUse { input, .. } => serde_json::to_string(input)
                .map(|s| s.len() / 4)
                .unwrap_or(100),
            ContentBlock::ToolResult { content, .. } => content.len() / 4,
            ContentBlock::ServerToolUse { .. }
            | ContentBlock::ToolSearchToolResult { .. }
            | ContentBlock::CodeExecutionToolResult { .. } => 0,
        })
        .sum::<usize>()
}

pub fn estimate_tokens(messages: &[Message]) -> usize {
    // Rough estimate: ~4 chars per token. DeepSeek thinking-mode rule: any
    // assistant message with tool_calls keeps its reasoning_content forever
    // (replayed in all subsequent requests). Final text-only answers drop it.
    messages
        .iter()
        .map(|message| estimate_tokens_for_message(message, message_has_tool_use(message)))
        .sum()
}

pub(crate) fn message_has_tool_use(message: &Message) -> bool {
    message
        .content
        .iter()
        .any(|block| matches!(block, ContentBlock::ToolUse { .. }))
}

pub(crate) fn estimate_text_tokens_conservative(text: &str) -> usize {
    text.chars().count().div_ceil(3)
}

pub(crate) fn estimate_system_tokens_conservative(system: Option<&SystemPrompt>) -> usize {
    match system {
        Some(SystemPrompt::Text(text)) => estimate_text_tokens_conservative(text),
        Some(SystemPrompt::Blocks(blocks)) => blocks
            .iter()
            .map(|block| estimate_text_tokens_conservative(&block.text))
            .sum(),
        None => 0,
    }
}

/// Conservative estimate for full request input tokens (messages + system + framing).
#[must_use]
pub fn estimate_input_tokens_conservative(
    messages: &[Message],
    system: Option<&SystemPrompt>,
) -> usize {
    let message_tokens = estimate_tokens(messages).saturating_mul(3).div_ceil(2);
    let system_tokens = estimate_system_tokens_conservative(system);
    let framing_overhead = messages.len().saturating_mul(12).saturating_add(48);
    message_tokens
        .saturating_add(system_tokens)
        .saturating_add(framing_overhead)
}

