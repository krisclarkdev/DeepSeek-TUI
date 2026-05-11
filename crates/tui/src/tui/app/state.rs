//! Application state for the `DeepSeek` TUI.

use std::collections::{HashMap, HashSet, VecDeque};
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use ratatui::layout::Rect;
use serde_json::Value;
use thiserror::Error;

use crate::artifacts::ArtifactRecord;
use crate::client::PromptInspection;
use crate::compaction::CompactionConfig;
use crate::config::{
    ApiProvider, Config, DEFAULT_TEXT_MODEL, SavedCredential, has_api_key, save_api_key,
};
use crate::config_ui::ConfigUiMode;
use crate::core::coherence::CoherenceState;
use crate::cycle_manager::{CycleBriefing, CycleConfig};
use crate::hooks::{HookContext, HookEvent, HookExecutor, HookResult};
use crate::localization::{Locale, MessageId, resolve_locale, tr};
use crate::models::{Message, SystemPrompt, compaction_threshold_for_model_and_effort};
use crate::palette::{self, UiTheme};
use crate::pricing::{CostCurrency, CostEstimate};
use crate::session_manager::SessionContextReference;
use crate::settings::Settings;
use crate::tools::plan::{SharedPlanState, new_shared_plan_state};
use crate::tools::shell::new_shared_shell_manager;
use crate::tools::spec::RuntimeToolServices;
use crate::tools::subagent::SubAgentResult;
use crate::tools::todo::{SharedTodoList, new_shared_todo_list};
use crate::tui::active_cell::ActiveCell;
use crate::tui::approval::ApprovalMode;
use crate::tui::clipboard::{ClipboardContent, ClipboardHandler};
use crate::tui::file_mention::ContextReference;
use crate::tui::history::{HistoryCell, TranscriptRenderOptions};
use crate::tui::paste_burst::{FlushResult, PasteBurst};
use crate::tui::scrolling::{MouseScrollState, TranscriptLineMeta, TranscriptScroll};
use crate::tui::selection::{SelectionAutoscroll, TranscriptSelection};
use crate::tui::streaming::StreamingState;
use crate::tui::transcript::TranscriptViewCache;
use crate::tui::views::ViewStack;

// === Types ===

/// State machine for onboarding new users.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OnboardingState {
    Welcome,
    /// Pick the UI locale before any other config decisions (#566).
    /// Defaults to auto-detection from `LC_ALL` / `LANG`; explicit picks
    /// land in `~/.deepseek/settings.toml` via `Settings::set("locale", …)`.
    Language,
    ApiKey,
    TrustDirectory,
    Tips,
    None,
}

pub(crate) fn initial_onboarding_state(
    skip_onboarding: bool,
    was_onboarded: bool,
    needs_api_key: bool,
    needs_workspace_trust: bool,
) -> OnboardingState {
    if skip_onboarding || (was_onboarded && !needs_api_key && !needs_workspace_trust) {
        return OnboardingState::None;
    }

    if was_onboarded && needs_api_key {
        OnboardingState::ApiKey
    } else if was_onboarded && needs_workspace_trust {
        OnboardingState::TrustDirectory
    } else {
        OnboardingState::Welcome
    }
}

pub(crate) fn onboarding_is_workspace_trust_gate(
    skip_onboarding: bool,
    was_onboarded: bool,
    needs_api_key: bool,
    needs_workspace_trust: bool,
) -> bool {
    !skip_onboarding && was_onboarded && !needs_api_key && needs_workspace_trust
}

/// Supported application modes for the TUI.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AppMode {
    Agent,
    Yolo,
    Plan,
}

/// One row in the per-turn cache-telemetry ring (`/cache` debug surface, #263).
#[derive(Debug, Clone)]
pub struct TurnCacheRecord {
    /// Provider-reported total input tokens for the turn (cache-hit +
    ///   cache-miss + uncategorized). Useful for sanity-checking that hits +
    ///   misses sum back to roughly the prompt size.
    pub input_tokens: u32,
    /// Provider-reported output tokens.
    pub output_tokens: u32,
    /// `prompt_cache_hit_tokens` from DeepSeek's usage payload. `None` when
    ///   the model in use does not report cache telemetry (see
    ///   `Capabilities::cache_telemetry_supported`).
    pub cache_hit_tokens: Option<u32>,
    /// `prompt_cache_miss_tokens`. `None` when the provider did not report it
    ///   — in that case the `/cache` formatter infers the miss as
    ///   `input_tokens − cache_hit_tokens`.
    pub cache_miss_tokens: Option<u32>,
    /// Approximate tokens spent re-sending prior `reasoning_content` on
    ///   V4-thinking tool-calling turns (chars/3 heuristic). Helps separate
    ///   cache misses caused by reasoning-replay churn from misses caused by
    ///   real prefix instability.
    pub reasoning_replay_tokens: Option<u32>,
    /// Local timestamp the turn telemetry was recorded.
    pub recorded_at: Instant,
}

/// DeepSeek reasoning-effort tier, mirrored on ChatGPT/Claude effort pickers.
///
/// The config file accepts all five string values for forward-compat with
/// providers that expose the full spectrum; DeepSeek currently collapses
/// `Low`/`Medium` → `high` and `Max` → `max` at the API boundary. The
/// keyboard cycler (Shift+Tab) walks only the three behaviorally distinct
/// tiers: `Off` → `High` → `Max` → `Off`.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub enum ReasoningEffort {
    Off,
    Low,
    Medium,
    High,
    Auto,
    #[default]
    Max,
}

impl ReasoningEffort {
    /// Parse a config-file string into an effort tier. Unknown values fall
    /// back to the default (`Max`) rather than erroring out.
    #[must_use]
    pub fn from_setting(value: &str) -> Self {
        match value.trim().to_ascii_lowercase().as_str() {
            "off" | "disabled" | "none" | "false" => Self::Off,
            "low" | "minimal" => Self::Low,
            "medium" | "mid" => Self::Medium,
            "high" => Self::High,
            "auto" | "automatic" => Self::Auto,
            "max" | "maximum" | "xhigh" => Self::Max,
            _ => Self::default(),
        }
    }

    /// Canonical lowercase label used for config storage and UI hints.
    #[must_use]
    pub fn as_setting(self) -> &'static str {
        match self {
            Self::Off => "off",
            Self::Low => "low",
            Self::Medium => "medium",
            Self::High => "high",
            Self::Auto => "auto",
            Self::Max => "max",
        }
    }

    /// Short label for the header chip.
    #[must_use]
    pub fn short_label(self) -> &'static str {
        match self {
            Self::Off => "off",
            Self::Low => "low",
            Self::Medium => "med",
            Self::High => "high",
            Self::Auto => "auto",
            Self::Max => "max",
        }
    }

    /// Value forwarded to the engine/client. `None` means "provider default"
    /// (for `Off` we still emit `"off"` so the client can inject
    /// `thinking = {"type": "disabled"}`).
    #[must_use]
    pub fn api_value(self) -> Option<&'static str> {
        Some(self.as_setting())
    }

    /// Cycle through the three behaviorally distinct tiers.
    #[must_use]
    pub fn cycle_next(self) -> Self {
        match self {
            Self::Off => Self::High,
            Self::Auto => Self::Off,
            Self::Low | Self::Medium | Self::High => Self::Max,
            Self::Max => Self::Off,
        }
    }
}

/// Sidebar content focus mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SidebarFocus {
    Auto,
    Plan,
    Todos,
    Tasks,
    Agents,
    Context,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ComposerDensity {
    Compact,
    Comfortable,
    Spacious,
}

impl ComposerDensity {
    #[must_use]
    pub fn from_setting(value: &str) -> Self {
        match value.trim().to_ascii_lowercase().as_str() {
            "compact" | "tight" => Self::Compact,
            "spacious" | "loose" => Self::Spacious,
            _ => Self::Comfortable,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TranscriptSpacing {
    Compact,
    Comfortable,
    Spacious,
}

impl TranscriptSpacing {
    #[must_use]
    pub fn from_setting(value: &str) -> Self {
        match value.trim().to_ascii_lowercase().as_str() {
            "compact" | "tight" => Self::Compact,
            "spacious" | "loose" => Self::Spacious,
            _ => Self::Comfortable,
        }
    }
}

impl SidebarFocus {
    #[must_use]
    pub fn from_setting(value: &str) -> Self {
        match value.trim().to_ascii_lowercase().as_str() {
            "plan" => Self::Plan,
            "todos" => Self::Todos,
            "tasks" => Self::Tasks,
            "agents" | "subagents" | "sub-agents" => Self::Agents,
            "context" | "session" => Self::Context,
            _ => Self::Auto,
        }
    }

    #[must_use]
    #[allow(dead_code)]
    pub fn as_setting(self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::Plan => "plan",
            Self::Todos => "todos",
            Self::Tasks => "tasks",
            Self::Agents => "agents",
            Self::Context => "context",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StatusToastLevel {
    Info,
    Success,
    Warning,
    Error,
}

#[derive(Debug, Clone)]
pub struct StatusToast {
    pub text: String,
    pub level: StatusToastLevel,
    pub created_at: Instant,
    pub ttl_ms: Option<u64>,
}

impl StatusToast {
    #[must_use]
    pub fn new(text: impl Into<String>, level: StatusToastLevel, ttl_ms: Option<u64>) -> Self {
        Self {
            text: text.into(),
            level,
            created_at: Instant::now(),
            ttl_ms,
        }
    }

    #[must_use]
    pub fn is_expired(&self, now: Instant) -> bool {
        self.ttl_ms
            .is_some_and(|ttl| now.duration_since(self.created_at).as_millis() >= u128::from(ttl))
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ComposerHistorySearch {
    pub(crate) pre_search_input: String,
    pub(crate) pre_search_cursor: usize,
    pub(crate) query: String,
    pub(crate) selected: usize,
}

impl ComposerHistorySearch {
    pub(crate) fn new(pre_search_input: String, pre_search_cursor: usize) -> Self {
        Self {
            pre_search_input,
            pre_search_cursor,
            query: String::new(),
            selected: 0,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct InputHistoryDraft {
    pub(crate) input: String,
    pub(crate) cursor: usize,
}

pub(crate) fn char_count(text: &str) -> usize {
    text.chars().count()
}

pub(crate) fn byte_index_at_char(text: &str, char_index: usize) -> usize {
    if char_index == 0 {
        return 0;
    }
    text.char_indices()
        .nth(char_index)
        .map(|(idx, _)| idx)
        .unwrap_or_else(|| text.len())
}

pub(crate) fn remove_char_at(text: &mut String, char_index: usize) -> bool {
    let start = byte_index_at_char(text, char_index);
    if start >= text.len() {
        return false;
    }
    let ch = text[start..].chars().next().unwrap();
    let end = start + ch.len_utf8();
    text.replace_range(start..end, "");
    true
}

pub(crate) fn normalize_paste_text(text: &str) -> String {
    if text.contains('\r') {
        text.replace("\r\n", "\n").replace('\r', "")
    } else {
        text.to_string()
    }
}

pub(crate) fn sanitize_api_key_text(text: &str) -> String {
    text.chars().filter(|c| !c.is_control()).collect()
}

pub(crate) const MAX_SUBMITTED_INPUT_CHARS: usize = 16_000;
pub(crate) const MAX_DRAFT_HISTORY: usize = 50;

impl AppMode {
    #[must_use]
    pub fn from_setting(value: &str) -> Self {
        match value.trim().to_ascii_lowercase().as_str() {
            "plan" => Self::Plan,
            "yolo" => Self::Yolo,
            _ => Self::Agent,
        }
    }

    #[must_use]
    pub fn as_setting(self) -> &'static str {
        match self {
            Self::Agent => "agent",
            Self::Yolo => "yolo",
            Self::Plan => "plan",
        }
    }

    /// Short label used in the UI footer.
    pub fn label(self) -> &'static str {
        match self {
            AppMode::Agent => "AGENT",
            AppMode::Yolo => "YOLO",
            AppMode::Plan => "PLAN",
        }
    }

    #[allow(dead_code)]
    /// Description shown in help or onboarding text.
    pub fn description(self) -> &'static str {
        match self {
            AppMode::Agent => "Agent mode - autonomous task execution with tools",
            AppMode::Yolo => "YOLO mode - full tool access without approvals",
            AppMode::Plan => "Plan mode - design before implementing",
        }
    }
}

/// Configuration required to bootstrap the TUI.
#[derive(Clone)]
#[allow(clippy::struct_excessive_bools)]
pub struct TuiOptions {
    pub model: String,
    pub workspace: PathBuf,
    pub config_path: Option<PathBuf>,
    pub config_profile: Option<String>,
    pub allow_shell: bool,
    /// Use the alternate screen buffer (fullscreen TUI).
    pub use_alt_screen: bool,
    /// Capture mouse input for internal scrolling/selection.
    pub use_mouse_capture: bool,
    /// Enable terminal bracketed-paste mode (OSC `?2004h` / `?2004l`). Defaults
    /// on; settable via `bracketed_paste = false` in `settings.toml` for the
    /// rare terminal that mishandles it.
    pub use_bracketed_paste: bool,
    /// Maximum number of concurrent sub-agents.
    pub max_subagents: usize,
    #[allow(dead_code)]
    pub skills_dir: PathBuf,
    #[allow(dead_code)]
    pub memory_path: PathBuf,
    #[allow(dead_code)]
    pub notes_path: PathBuf,
    #[allow(dead_code)]
    pub mcp_config_path: PathBuf,
    #[allow(dead_code)]
    pub use_memory: bool,
    /// Start in agent mode (defaults to agent; --yolo starts in YOLO)
    pub start_in_agent_mode: bool,
    /// Skip onboarding screens
    pub skip_onboarding: bool,
    /// Auto-approve tool executions (yolo mode)
    pub yolo: bool,
    /// Resume a previous session by ID
    pub resume_session_id: Option<String>,
    /// Pre-populate the composer with this text when the TUI starts.
    /// Used by `deepseek pr <N>` (#451) to drop the model into a
    /// session with the PR context already typed — the user can edit
    /// before sending or hit Enter to fire as-is.
    pub initial_input: Option<String>,
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct YoloRestoreState {
    pub(crate) allow_shell: bool,
    pub(crate) trust_mode: bool,
    pub(crate) approval_mode: ApprovalMode,
}

// === Sub-state structs for App field organization (#377) ===

/// Vim modal editing mode for the composer input area.
///
/// Enabled via `[composer] mode = "vim"` in `settings.toml`.  When the
/// composer vim mode is active the user starts in `Normal` mode and presses
/// `i`, `a`, or `o` to enter `Insert` mode.  `Esc` from `Insert` returns to
/// `Normal`.  Standard vim motions (`h`/`j`/`k`/`l`, `w`/`b`, `0`/`$`, `x`,
/// `dd`) work in `Normal` mode.  `Visual` is reserved for future selection
/// support and currently behaves like `Normal`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum VimMode {
    /// Normal / command mode — motions and operators, no text insertion.
    #[default]
    Normal,
    /// Insert mode — characters are appended at the cursor as typed.
    Insert,
    /// Visual mode — reserved for future selection support.
    Visual,
}

impl VimMode {
    /// Short status-bar label shown in the composer border.
    #[must_use]
    pub fn label(self) -> &'static str {
        match self {
            Self::Normal => "-- NORMAL --",
            Self::Insert => "-- INSERT --",
            Self::Visual => "-- VISUAL --",
        }
    }
}

/// Cached @-mention completion results to avoid re-walking the filesystem when
/// the cursor moves inside the same mention token.
#[derive(Debug, Clone)]
pub struct MentionCompletionCache {
    /// Workspace root used for this completion walk.
    pub workspace: PathBuf,
    /// Process cwd captured for cwd-relative completion entries.
    pub cwd: Option<PathBuf>,
    /// The partial text after `@` that triggered this completion.
    pub partial: String,
    /// Candidate limit used for this completion walk.
    pub limit: usize,
    /// Cached completion entries.
    pub entries: Vec<String>,
}

/// Composer input state — grouped fields for the text input area.
pub struct ComposerState {
    /// Current composer text content.
    pub input: String,
    /// Cursor position within `input` (in characters).
    pub cursor_position: usize,
    /// Single-entry kill buffer for emacs-style `Ctrl+K` cut / `Ctrl+Y` yank.
    pub kill_buffer: String,
    pub paste_burst: PasteBurst,
    pub input_history: Vec<String>,
    pub draft_history: VecDeque<String>,
    pub history_index: Option<usize>,
    pub(crate) history_navigation_draft: Option<InputHistoryDraft>,
    pub composer_history_search: Option<ComposerHistorySearch>,
    pub selected_attachment_index: Option<usize>,
    pub slash_menu_selected: usize,
    pub slash_menu_hidden: bool,
    pub mention_menu_selected: usize,
    pub mention_menu_hidden: bool,
    /// Cached @-mention completions to avoid re-walking the filesystem when
    /// the cursor moves inside the same mention token.
    pub mention_completion_cache: Option<MentionCompletionCache>,
    /// Whether vim modal editing is enabled for this composer.
    /// Sourced from `Settings::composer_vim_mode` at startup.
    pub vim_enabled: bool,
    /// Current vim editing mode.  Only meaningful when `vim_enabled` is true.
    pub vim_mode: VimMode,
    /// Pending `d` prefix for the `dd` delete-line operator.  Set when the
    /// user presses `d` in Normal mode; cleared on the next key (either `d`
    /// to complete `dd`, or any other key to cancel).
    pub vim_pending_d: bool,
}

impl Default for ComposerState {
    fn default() -> Self {
        Self {
            input: String::new(),
            cursor_position: 0,
            kill_buffer: String::new(),
            paste_burst: PasteBurst::default(),
            input_history: Vec::new(),
            draft_history: VecDeque::new(),
            history_index: None,
            history_navigation_draft: None,
            composer_history_search: None,
            selected_attachment_index: None,
            slash_menu_selected: 0,
            slash_menu_hidden: false,
            mention_menu_selected: 0,
            mention_menu_hidden: false,
            mention_completion_cache: None,
            vim_enabled: false,
            vim_mode: VimMode::Normal,
            vim_pending_d: false,
        }
    }
}

/// Viewport/scroll state — fields related to transcript scrolling and caching.
pub struct ViewportState {
    pub transcript_scroll: TranscriptScroll,
    pub pending_scroll_delta: i32,
    pub mouse_scroll: MouseScrollState,
    pub transcript_cache: TranscriptViewCache,
    pub transcript_selection: TranscriptSelection,
    pub selection_autoscroll: Option<SelectionAutoscroll>,
    pub transcript_scrollbar_dragging: bool,
    pub last_transcript_area: Option<Rect>,
    pub last_transcript_top: usize,
    pub last_transcript_visible: usize,
    pub last_transcript_total: usize,
    pub last_transcript_padding_top: usize,
    pub jump_to_latest_button_area: Option<Rect>,
}

impl Default for ViewportState {
    fn default() -> Self {
        Self {
            transcript_scroll: TranscriptScroll::to_bottom(),
            pending_scroll_delta: 0,
            mouse_scroll: MouseScrollState::new(),
            transcript_cache: TranscriptViewCache::new(),
            transcript_selection: TranscriptSelection::default(),
            selection_autoscroll: None,
            transcript_scrollbar_dragging: false,
            last_transcript_area: None,
            last_transcript_top: 0,
            last_transcript_visible: 0,
            last_transcript_total: 0,
            last_transcript_padding_top: 0,
            jump_to_latest_button_area: None,
        }
    }
}

/// Goal mode state (#397).
#[derive(Debug, Clone, Default)]
pub struct GoalState {
    pub goal_objective: Option<String>,
    pub goal_token_budget: Option<u32>,
    pub goal_started_at: Option<Instant>,
}

/// Session cost and token telemetry state.
#[derive(Debug, Clone)]
pub struct SessionState {
    pub session_cost: f64,
    pub session_cost_cny: f64,
    pub subagent_cost: f64,
    pub subagent_cost_cny: f64,
    pub subagent_cost_event_seqs: HashSet<u64>,
    pub displayed_cost_high_water: f64,
    pub displayed_cost_high_water_cny: f64,
    pub last_prompt_tokens: Option<u32>,
    pub last_completion_tokens: Option<u32>,
    pub last_prompt_cache_hit_tokens: Option<u32>,
    pub last_prompt_cache_miss_tokens: Option<u32>,
    pub last_reasoning_replay_tokens: Option<u32>,
    pub total_tokens: u32,
    pub total_conversation_tokens: u32,
    pub turn_cache_history: VecDeque<TurnCacheRecord>,
    pub last_cache_inspection: Option<PromptInspection>,
}

impl Default for SessionState {
    fn default() -> Self {
        Self {
            session_cost: 0.0,
            session_cost_cny: 0.0,
            subagent_cost: 0.0,
            subagent_cost_cny: 0.0,
            subagent_cost_event_seqs: HashSet::new(),
            displayed_cost_high_water: 0.0,
            displayed_cost_high_water_cny: 0.0,
            last_prompt_tokens: None,
            last_completion_tokens: None,
            last_prompt_cache_hit_tokens: None,
            last_prompt_cache_miss_tokens: None,
            last_reasoning_replay_tokens: None,
            total_tokens: 0,
            total_conversation_tokens: 0,
            turn_cache_history: VecDeque::new(),
            last_cache_inspection: None,
        }
    }
}

/// Global UI state for the TUI.
#[allow(clippy::struct_excessive_bools)]
pub struct App {
    pub mode: AppMode,
    /// Composer sub-state (input, cursor, history, menus).
    pub composer: ComposerState,
    /// Viewport sub-state (scroll, cache, selection).
    pub viewport: ViewportState,
    /// Goal sub-state.
    pub goal: GoalState,
    /// Session sub-state (cost, tokens, telemetry).
    pub session: SessionState,
    pub history: Vec<HistoryCell>,
    pub history_version: u64,
    /// Per-cell revision counter, kept in lockstep with `history`.
    pub history_revisions: Vec<u64>,
    /// Monotonic counter used to issue fresh per-cell revisions.
    pub next_history_revision: u64,
    pub api_messages: Vec<Message>,
    pub is_loading: bool,
    /// Degraded connectivity mode; new user inputs are queued for later retry.
    pub offline_mode: bool,
    /// Whether an `EngineEvent::Error` has already been posted for the
    /// current turn. Suppresses the redundant "Turn failed:" status line
    /// that `TurnComplete { error: .. }` would otherwise emit on top of
    /// the in-transcript error cell.
    pub turn_error_posted: bool,
    /// Legacy status text sink retained for compatibility with existing call sites.
    pub status_message: Option<String>,
    /// Recent status toasts (ephemeral, newest at back).
    pub status_toasts: VecDeque<StatusToast>,
    /// Sticky status toast used for important warnings/errors.
    pub sticky_status: Option<StatusToast>,
    /// Last status text already promoted from `status_message` into toast state.
    pub last_status_message_seen: Option<String>,
    pub model: String,
    /// When true, the model is auto-selected based on request complexity
    /// rather than using a fixed model. The `/model auto` command sets this.
    /// `dispatch_user_message` calls `auto_model_heuristic` to resolve the
    /// effective model for each outbound message.
    pub auto_model: bool,
    /// Last concrete model chosen while `auto_model` is active.
    pub last_effective_model: Option<String>,
    /// Current API provider (mirrors `Config::api_provider`).
    /// Updated by `/provider` switches so the UI/commands can read the
    /// active backend without re-deriving it from the live config.
    pub api_provider: ApiProvider,
    /// Current reasoning-effort tier for DeepSeek thinking mode.
    /// Cycled via Shift+Tab; initialized from config at startup.
    pub reasoning_effort: ReasoningEffort,
    /// Last concrete thinking tier chosen while `reasoning_effort` is auto.
    pub last_effective_reasoning_effort: Option<ReasoningEffort>,
    pub workspace: PathBuf,
    pub config_path: Option<PathBuf>,
    pub config_profile: Option<String>,
    pub mcp_config_path: PathBuf,
    pub skills_dir: PathBuf,
    /// Path to the user-memory file (#489). Always populated; only
    /// consulted when `use_memory` is `true`.
    pub memory_path: PathBuf,
    /// Whether the user-memory feature is enabled (#489). Mirrors
    /// `Config::memory_enabled()` at app boot. Used by the `# foo`
    /// composer interception, the `/memory` slash command, and tool
    /// registration for `remember`.
    pub use_memory: bool,
    pub use_alt_screen: bool,
    pub use_mouse_capture: bool,
    /// When true, plain Up/Down on an empty composer scroll the transcript
    /// instead of navigating input history (#1117 opt-in).
    pub composer_arrows_scroll: bool,
    pub use_bracketed_paste: bool,
    pub use_paste_burst_detection: bool,
    /// Set to `true` the first time a real `Event::Paste` arrives during a
    /// session. Once set, `handle_paste_burst_key` short-circuits — there's
    /// no point running the rapid-keypress heuristic on a terminal that
    /// already delivers paste-as-event correctly. Avoids paste-burst false
    /// positives on Ghostty / iTerm2 / WezTerm / Windows Terminal where
    /// fast typing or IME commits could otherwise be mis-classified as a
    /// paste burst (#1322 follow-up).
    pub bracketed_paste_seen: bool,
    #[allow(dead_code)]
    pub system_prompt: Option<SystemPrompt>,
    pub auto_compact: bool,
    pub calm_mode: bool,
    pub low_motion: bool,
    /// Pending #61 (animated working strip). Set from config but not read
    /// until the footer widget consumes it.
    #[allow(dead_code)]
    pub fancy_animations: bool,
    pub show_thinking: bool,
    pub verbose_transcript: bool,
    pub show_tool_details: bool,
    pub ui_locale: Locale,
    pub cost_currency: CostCurrency,
    pub composer_density: ComposerDensity,
    pub composer_border: bool,
    pub transcript_spacing: TranscriptSpacing,
    pub sidebar_width_percent: u16,
    pub sidebar_focus: SidebarFocus,
    /// Whether the session-context panel is enabled (#504).
    pub context_panel: bool,
    /// File-tree pane state. `None` when hidden; `Some` when visible.
    pub file_tree: Option<crate::tui::file_tree::FileTreeState>,
    #[allow(dead_code)]
    pub compact_threshold: usize,
    pub max_input_history: usize,
    pub allow_shell: bool,
    pub max_subagents: usize,
    /// Cached sub-agent snapshots for UI views.
    pub subagent_cache: Vec<SubAgentResult>,
    /// Last known per-agent progress text for running sub-agents.
    pub agent_progress: HashMap<String, String>,
    /// In-transcript sub-agent card index by `agent_id` (issue #128).
    /// Maps each live sub-agent to the `HistoryCell::SubAgent` it renders
    /// into, so successive mailbox envelopes mutate the same cell rather
    /// than spawning duplicates.
    pub subagent_card_index: HashMap<String, usize>,
    /// History index of the most recent FanoutCard. Sibling sub-agents
    /// spawned by the same `rlm` invocation route into this card; reset
    /// when a fresh fanout-family tool call starts.
    pub last_fanout_card_index: Option<usize>,
    /// Most recently observed sub-agent dispatch tool name (set on
    /// `ToolCallStarted` for `agent_spawn` / `rlm` / etc., cleared
    /// after the first `Started` mailbox envelope routes through it).
    pub pending_subagent_dispatch: Option<String>,
    /// Animation anchor for status-strip active sub-agent spinner.
    pub agent_activity_started_at: Option<Instant>,
    pub ui_theme: UiTheme,
    // Onboarding
    pub onboarding: OnboardingState,
    pub onboarding_needs_api_key: bool,
    pub onboarding_workspace_trust_gate: bool,
    pub api_key_env_only: bool,
    pub api_key_input: String,
    pub api_key_cursor: usize,
    // Hooks system
    pub hooks: HookExecutor,
    #[allow(dead_code)]
    pub yolo: bool,
    pub(crate) yolo_restore: Option<YoloRestoreState>,
    // Clipboard handler
    pub clipboard: ClipboardHandler,
    // Tool approval session allowlist
    pub approval_session_approved: HashSet<String>,
    /// Approval keys (or tool names) the user has denied or aborted in
    /// this session. Subsequent re-requests for the same approval key
    /// auto-deny without re-prompting (#360) — the model can retry a
    /// dangerous command after being told no, but the user shouldn't
    /// have to keep dismissing the same dialog.
    pub approval_session_denied: HashSet<String>,
    pub approval_mode: ApprovalMode,
    // Modal view stack (approval/help/etc.)
    pub view_stack: ViewStack,
    /// Esc-Esc backtrack state machine (#133). `Inactive` by default; first
    /// Esc primes, second Esc opens the live-transcript overlay scoped to
    /// previous user messages so the user can rewind a turn.
    pub backtrack: crate::tui::backtrack::BacktrackState,
    /// Current session ID for auto-save updates
    pub current_session_id: Option<String>,
    /// Metadata-only registry of large tool outputs produced in this session.
    pub session_artifacts: Vec<ArtifactRecord>,
    /// Trust mode - allow access outside workspace
    pub trust_mode: bool,
    /// Ordered list of footer items the user wants visible. Sourced from
    /// `tui.status_items` in `~/.deepseek/config.toml` at startup; mutated
    /// live by `/statusline`. The renderer iterates this slice; no item is
    /// hardcoded in the footer code path.
    pub status_items: Vec<crate::config::StatusItem>,
    /// Project documentation (AGENTS.md or CLAUDE.md)
    #[allow(dead_code)]
    pub project_doc: Option<String>,
    /// Plan state for tracking tasks
    pub plan_state: SharedPlanState,
    /// Whether a plan follow-up prompt is waiting for user input
    pub plan_prompt_pending: bool,
    /// Whether update_plan was called during the current turn
    pub plan_tool_used_in_turn: bool,
    /// Todo list for `TodoWriteTool`
    #[allow(dead_code)] // For future engine integration
    pub todos: SharedTodoList,
    /// Durable runtime services exposed to model-visible task/automation tools.
    pub runtime_services: RuntimeToolServices,
    /// Last MCP manager/discovery snapshot shown in the UI.
    pub mcp_snapshot: Option<crate::mcp::McpManagerSnapshot>,
    /// Number of MCP servers declared in the user's config at app boot.
    /// Used by the footer chip (#502) so a count is visible even before
    /// the user runs `/mcp` for the first time. `0` hides the chip.
    pub mcp_configured_count: usize,
    /// Set after in-TUI MCP config edits because the engine caches its MCP pool.
    pub mcp_restart_required: bool,
    /// Tool execution log
    pub tool_log: Vec<String>,
    /// Active skill to apply to next user message
    pub active_skill: Option<String>,
    /// Cached (name, description) pairs from the skill registry.
    /// Populated once at startup and refreshed on install/uninstall so
    /// the slash menu can show skills without filesystem I/O on every keystroke.
    pub cached_skills: Vec<(String, String)>,
    /// Tool call cells by tool id (for cells already finalized in `history`).
    /// While a tool call is in flight inside `active_cell`, it is tracked by
    /// `active_tool_entries` instead and migrated here at flush time.
    pub tool_cells: HashMap<String, usize>,
    /// Full tool input/output keyed by history cell index.
    pub tool_details_by_cell: HashMap<usize, ToolDetailRecord>,
    /// Linked context references keyed by the visible user history cell that
    /// introduced them.
    pub context_references_by_cell: HashMap<usize, Vec<SessionContextReference>>,
    /// Session-wide context references persisted with saved sessions.
    pub session_context_references: Vec<SessionContextReference>,
    /// In-flight tool/exec group for the current turn. Mutated in place as
    /// parallel tool calls start and complete; flushed into `history` on
    /// `TurnComplete`.
    pub active_cell: Option<ActiveCell>,
    /// Revision counter for `active_cell`. Combined with `active_cell.revision`
    /// when feeding the transcript cache so cached lines for the synthetic
    /// active-cell row are invalidated on every mutation.
    pub active_cell_revision: u64,
    /// Pending tool details for entries that live inside `active_cell`.
    /// Keyed by tool id rather than cell index because the active cell's
    /// virtual index can shift (orphan completions push real cells in
    /// between). Migrated into `tool_details_by_cell` on flush.
    pub active_tool_details: HashMap<String, ToolDetailRecord>,
    /// Active exploring cell entry index (within `active_cell.entries`).
    /// `None` once the active cell flushes or no exploring entry exists.
    pub exploring_cell: Option<usize>,
    /// Mapping of exploring tool ids to `(entry index in active_cell, entry
    /// within ExploringCell)`. Used to update individual exploring entries
    /// when their tools complete.
    pub exploring_entries: HashMap<String, (usize, usize)>,
    /// Tool calls that should be ignored by the UI
    pub ignored_tool_calls: HashSet<String>,
    /// Last exec wait command shown (for duplicate suppression)
    pub last_exec_wait_command: Option<String>,
    /// Current streaming assistant cell
    pub streaming_message_index: Option<usize>,
    /// Index into `active_cell.entries` of the thinking entry currently being
    /// streamed. `None` when no thinking block is in flight. P2.3 routes
    /// thinking into the active cell so it groups visually with tool calls
    /// until the next assistant prose chunk flushes the group into history.
    pub streaming_thinking_active_entry: Option<usize>,
    /// Newline-gated streaming collector state.
    pub streaming_state: StreamingState,
    /// Accumulated reasoning text
    pub reasoning_buffer: String,
    /// Live reasoning header extracted from bold text
    pub reasoning_header: Option<String>,
    /// Last completed reasoning block
    pub last_reasoning: Option<String>,
    /// Tool calls captured for the pending assistant message
    pub pending_tool_uses: Vec<(String, String, Value)>,
    /// User messages queued while a turn is running
    pub queued_messages: VecDeque<QueuedMessage>,
    /// Draft queued message being edited
    pub queued_draft: Option<QueuedMessage>,
    /// Legacy pending-steer bucket retained for session compatibility. New
    /// in-flight input uses Enter for same-turn steering and Tab for queued
    /// follow-ups; Esc only cancels the active turn.
    pub pending_steers: VecDeque<QueuedMessage>,
    /// Engine-rejected steers (e.g. a tool was already running and couldn't be
    /// cancelled cleanly). Surfaced in the pending-input preview so the user
    /// knows the steer was deferred to end-of-turn. Today no engine path
    /// produces these; the field is scaffolding for a future signalling
    /// channel and the bucket renders identically when populated.
    pub rejected_steers: VecDeque<String>,
    /// Legacy resend flag for pending steer recovery.
    pub submit_pending_steers_after_interrupt: bool,
    /// Start time for current turn
    pub turn_started_at: Option<Instant>,
    /// Sum of completed turn durations for this `App` instance (#448
    /// follow-up). Drives the footer's `worked Nh Mm` chip so the
    /// label reflects actual model work, not wall-clock since launch.
    /// Incremented on `TurnComplete` from the elapsed time of the
    /// just-finished turn. Resets per launch.
    pub cumulative_turn_duration: std::time::Duration,
    /// Current runtime turn id (if known).
    pub runtime_turn_id: Option<String>,
    /// Current runtime turn status (if known).
    pub runtime_turn_status: Option<String>,
    /// When the UI accepted a user message but has not observed `TurnStarted` yet.
    pub dispatch_started_at: Option<Instant>,

    /// Cached git context snapshot for the footer.
    pub workspace_context: Option<String>,
    /// Shared cell for async git context updates (#399 S1).
    pub workspace_context_cell: std::sync::Arc<std::sync::Mutex<Option<String>>>,
    /// Timestamp for cached workspace context.
    pub workspace_context_refreshed_at: Option<Instant>,
    /// Cached background tasks for sidebar rendering.
    pub task_panel: Vec<TaskPanelEntry>,
    /// Whether the UI needs to be redrawn.
    pub needs_redraw: bool,
    /// When the current thinking block started (for duration tracking).
    pub thinking_started_at: Option<Instant>,
    /// Whether context compaction is currently in progress.
    pub is_compacting: bool,
    /// Set when the user scrolls up/down during a streaming turn so subsequent
    /// streamed chunks don't yank the view back to the live tail. Cleared
    /// when the user explicitly returns to bottom or the turn completes.
    pub user_scrolled_during_stream: bool,
    /// Plain-language session coherence state for the footer.
    pub coherence_state: CoherenceState,
    /// Timestamp of the last user message send (for brief visual feedback).
    pub last_send_at: Option<Instant>,
    /// Two-tap quit confirmation. When set, a prior Ctrl+C in idle state has
    /// armed the quit shortcut; a second Ctrl+C before this `Instant` exits
    /// the app, while expiry silently re-arms the prompt for next time.
    /// Stays `None` while a turn is in flight or a modal/picker is open so
    /// Ctrl+C keeps its current "interrupt this turn" semantics in those
    /// states. See [`App::arm_quit`] / [`App::quit_is_armed`].
    pub quit_armed_until: Option<Instant>,

    /// Number of checkpoint-restart cycles crossed in this session
    /// (issue #124). Mirrors `Session.cycle_count` on the engine side.
    pub cycle_count: u32,

    /// Briefings produced at past cycle boundaries, in chronological order.
    /// Used by `/cycles` and `/cycle <n>` slash commands.
    pub cycle_briefings: Vec<CycleBriefing>,

    /// Active cycle configuration (token threshold, briefing cap, per-model
    /// overrides). Loaded from config and forwarded to the engine.
    pub cycle: CycleConfig,

    // === Goal Mode (#397) ===
    /// Transcript cells the user has collapsed (hidden from view).
    /// Stores **original** virtual cell indices (pre-filtering).
    pub collapsed_cells: HashSet<usize>,
    /// Mapping from filtered cell index → original virtual index.
    /// Populated during `ChatWidget::new` by filtering out collapsed cells.
    /// Used by `build_context_menu_entries` to convert line-meta indices
    /// back to original indices for the `HideCell` / `ShowCell` actions.
    pub collapsed_cell_map: Vec<usize>,

    /// Whether `/edit` has loaded the last user message into the composer and
    /// the next submit should replace (not append to) the last exchange.
    pub edit_in_progress: bool,

    /// Whether LSP diagnostics are currently enabled. Mirrors the config file
    /// `[lsp].enabled` setting. Toggled at runtime via `/lsp on|off`.
    pub lsp_enabled: bool,
}

/// Message queued while the engine is busy.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct QueuedMessage {
    pub display: String,
    pub skill_instruction: Option<String>,
}

/// How a freshly-typed user input should be sent.
///
/// Picked by [`App::decide_submit_disposition`] when the user hits Enter on a
/// non-empty composer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SubmitDisposition {
    /// Engine idle and online: send immediately.
    Immediate,
    /// Park on `queued_messages` (offline, or engine busy — #382).
    Queue,
    /// Explicit steer via Ctrl+Enter (#382). Not returned by `decide_submit_disposition`.
    #[allow(dead_code)]
    Steer,
    /// Park on `queued_messages` for dispatch after TurnComplete.
    /// Legacy path; #382 unified busy states under `Queue`.
    #[allow(dead_code)]
    QueueFollowUp,
}

/// Detailed tool payload attached to a history cell.
#[derive(Debug, Clone)]
pub struct ToolDetailRecord {
    pub tool_id: String,
    pub tool_name: String,
    pub input: Value,
    pub output: Option<String>,
}

/// Lightweight task view for sidebar rendering.
#[derive(Debug, Clone)]
pub struct TaskPanelEntry {
    pub id: String,
    pub status: String,
    pub prompt_summary: String,
    pub duration_ms: Option<u64>,
}

impl QueuedMessage {
    pub fn new(display: String, skill_instruction: Option<String>) -> Self {
        Self {
            display,
            skill_instruction,
        }
    }

    #[allow(dead_code)] // Tests and queue helpers use the display-only form; send path resolves @mentions.
    pub fn content(&self) -> String {
        if let Some(skill_instruction) = self.skill_instruction.as_ref() {
            format!(
                "{skill_instruction}\n\n---\n\nUser request: {}",
                self.display
            )
        } else {
            self.display.clone()
        }
    }
}

// === Errors ===

/// Errors that can occur while submitting API keys during onboarding.
#[derive(Debug, Error)]
pub enum ApiKeyError {
    /// The provided API key was empty.
    #[error("Failed to save API key: API key cannot be empty")]
    Empty,
    /// Persisting the API key failed.
    #[error("Failed to save API key: {source}")]
    SaveFailed { source: anyhow::Error },
}

// === Deref to ComposerState for backward compat ===

impl std::ops::Deref for App {
    type Target = ComposerState;
    fn deref(&self) -> &Self::Target {
        &self.composer
    }
}

impl std::ops::DerefMut for App {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.composer
    }
}

// === App State ===

