use std::collections::{HashMap, HashSet, VecDeque};
use std::path::Path;
use std::time::{Duration, Instant};


use crate::compaction::CompactionConfig;
use crate::config::{
    ApiProvider, Config, DEFAULT_TEXT_MODEL, SavedCredential, has_api_key, save_api_key,
};
use crate::core::coherence::CoherenceState;
use crate::cycle_manager::CycleConfig;
use crate::hooks::{HookContext, HookEvent, HookExecutor, HookResult};
use crate::localization::{MessageId, resolve_locale, tr};
use crate::models::compaction_threshold_for_model_and_effort;
use crate::palette::{self};
use crate::pricing::{CostCurrency, CostEstimate};
use crate::session_manager::SessionContextReference;
use crate::settings::Settings;
use crate::tools::plan::new_shared_plan_state;
use crate::tools::shell::new_shared_shell_manager;
use crate::tools::spec::RuntimeToolServices;
use crate::tools::todo::new_shared_todo_list;
use crate::tui::active_cell::ActiveCell;
use crate::tui::approval::ApprovalMode;
use crate::tui::clipboard::{ClipboardContent, ClipboardHandler};
use crate::tui::file_mention::ContextReference;
use crate::tui::history::{HistoryCell, TranscriptRenderOptions};
use crate::tui::paste_burst::{FlushResult, PasteBurst};
use crate::tui::scrolling::{TranscriptLineMeta, TranscriptScroll};
use crate::tui::streaming::StreamingState;
use crate::tui::transcript::TranscriptViewCache;
use crate::tui::views::ViewStack;

use super::state::*;
use super::actions::*;
impl App {
    /// Cap on the session turn-cache history. Holds enough turns to debug a long
    /// session without being so large the on-screen `/cache` table wraps.
    pub const TURN_CACHE_HISTORY_CAP: usize = 50;

    /// Append a per-turn cache-telemetry record, trimming the oldest entry once
    /// the ring exceeds [`Self::TURN_CACHE_HISTORY_CAP`].
    pub fn push_turn_cache_record(&mut self, record: TurnCacheRecord) {
        self.session.turn_cache_history.push_back(record);
        while self.session.turn_cache_history.len() > Self::TURN_CACHE_HISTORY_CAP {
            self.session.turn_cache_history.pop_front();
        }
    }

    pub(crate) fn clear_model_scoped_telemetry(&mut self) {
        self.session.last_prompt_tokens = None;
        self.session.last_completion_tokens = None;
        self.session.last_prompt_cache_hit_tokens = None;
        self.session.last_prompt_cache_miss_tokens = None;
        self.session.last_reasoning_replay_tokens = None;
        self.session.turn_cache_history.clear();
    }

    pub fn tr(&self, id: MessageId) -> &'static str {
        tr(self.ui_locale, id)
    }

    #[allow(clippy::too_many_lines)]
    pub fn new(options: TuiOptions, config: &Config) -> Self {
        let TuiOptions {
            model,
            workspace,
            config_path,
            config_profile,
            allow_shell,
            use_alt_screen,
            use_mouse_capture,
            use_bracketed_paste,
            max_subagents,
            skills_dir: global_skills_dir,
            memory_path,
            notes_path: _,
            mcp_config_path,
            use_memory,
            start_in_agent_mode,
            skip_onboarding,
            yolo,
            resume_session_id: _,
            initial_input,
        } = options;

        let mut provider = config.api_provider();

        // Check if API key exists
        let needs_api_key = !has_api_key(config);
        let api_key_env_only = crate::config::active_provider_uses_env_only_api_key(config);
        let was_onboarded = crate::tui::onboarding::is_onboarded();
        let settings = Settings::load().unwrap_or_else(|_| Settings::default());

        // Let settings override the config provider so runtime switches survive restarts.
        if let Some(ref provider_str) = settings.default_provider
            && let Some(parsed) = ApiProvider::parse(provider_str)
        {
            provider = parsed;
        }
        let auto_compact = settings.auto_compact;
        let calm_mode = settings.calm_mode;
        let low_motion = settings.low_motion;
        let fancy_animations = settings.fancy_animations;
        let show_thinking = settings.show_thinking;
        let show_tool_details = settings.show_tool_details;
        let ui_locale = resolve_locale(&settings.locale);
        let cost_currency =
            CostCurrency::from_setting(&settings.cost_currency).unwrap_or(CostCurrency::Usd);
        let composer_density = ComposerDensity::from_setting(&settings.composer_density);
        let composer_border = settings.composer_border;
        let composer_vim_enabled = settings
            .composer_vim_mode
            .trim()
            .eq_ignore_ascii_case("vim");
        let transcript_spacing = TranscriptSpacing::from_setting(&settings.transcript_spacing);
        let sidebar_width_percent = settings.sidebar_width_percent;
        let sidebar_focus = SidebarFocus::from_setting(&settings.sidebar_focus);
        let max_input_history = settings.max_input_history;
        let use_paste_burst_detection = settings.paste_burst_detection;
        let mut ui_theme = palette::UiTheme::detect();
        if let Some(background) = settings
            .background_color
            .as_deref()
            .and_then(palette::parse_hex_rgb_color)
        {
            ui_theme = ui_theme.with_background_color(background);
        }
        let model = settings
            .provider_models
            .as_ref()
            .and_then(|m| m.get(provider.as_str()).cloned())
            .or_else(|| {
                // default_model is a DeepSeek-centric setting; other providers
                // get their model from config.toml / env (e.g. OPENAI_MODEL).
                if matches!(provider, ApiProvider::Deepseek | ApiProvider::DeepseekCN) {
                    settings.default_model.clone()
                } else {
                    None
                }
            })
            .unwrap_or(model);
        let auto_model = model.trim().eq_ignore_ascii_case("auto");
        let threshold_model = if auto_model {
            DEFAULT_TEXT_MODEL
        } else {
            model.as_str()
        };
        let compact_threshold =
            compaction_threshold_for_model_and_effort(threshold_model, config.reasoning_effort());
        let reasoning_effort = if auto_model {
            ReasoningEffort::Auto
        } else {
            config
                .reasoning_effort()
                .map_or_else(ReasoningEffort::default, |s| {
                    ReasoningEffort::from_setting(s)
                })
        };

        // Start in YOLO mode if --yolo flag was passed
        let preferred_mode = AppMode::from_setting(&settings.default_mode);
        let initial_mode = if yolo {
            AppMode::Yolo
        } else if start_in_agent_mode {
            AppMode::Agent
        } else {
            preferred_mode
        };
        let needs_workspace_trust =
            initial_mode != AppMode::Yolo && crate::tui::onboarding::needs_trust(&workspace);
        let onboarding = initial_onboarding_state(
            skip_onboarding,
            was_onboarded,
            needs_api_key,
            needs_workspace_trust,
        );
        let onboarding_workspace_trust_gate = onboarding_is_workspace_trust_gate(
            skip_onboarding,
            was_onboarded,
            needs_api_key,
            needs_workspace_trust,
        );

        let yolo_restore = if initial_mode == AppMode::Yolo {
            Some(YoloRestoreState {
                allow_shell: config.allow_shell(),
                trust_mode: false,
                approval_mode: config
                    .approval_policy
                    .as_deref()
                    .and_then(ApprovalMode::from_config_value)
                    .unwrap_or_default(),
            })
        } else {
            None
        };
        let allow_shell = allow_shell || initial_mode == AppMode::Yolo;
        let shell_manager = new_shared_shell_manager(workspace.clone());

        // Initialize hooks executor from config
        let hooks_config = config.hooks_config();
        let hooks = HookExecutor::new(hooks_config, workspace.clone());

        // Initialize plan state
        let plan_state = new_shared_plan_state();

        let agents_skills_dir = workspace.join(".agents").join("skills");
        let local_skills_dir = workspace.join("skills");
        let agents_global_skills_dir = crate::skills::agents_global_skills_dir();
        let skills_dir = if agents_skills_dir.exists() {
            agents_skills_dir
        } else if local_skills_dir.exists() {
            local_skills_dir
        } else if config.skills_dir.is_none()
            && let Some(global_agents) = agents_global_skills_dir
            && global_agents.exists()
        {
            global_agents
        } else {
            global_skills_dir
        };
        let cached_skills = Self::discover_cached_skills(&workspace);

        let input_history = crate::composer_history::load_history();
        let (initial_input_text, initial_input_cursor) = match initial_input {
            // #451: pre-populate the composer when invoked via
            // `deepseek pr <N>` (or any future caller that wants to
            // drop the model into a session with context already
            // typed). Cursor lands at the end so Enter sends as-is.
            Some(text) if !text.is_empty() => {
                let cursor = text.len();
                (text, cursor)
            }
            _ => (String::new(), 0),
        };
        Self {
            mode: initial_mode,
            composer: ComposerState {
                input: initial_input_text,
                cursor_position: initial_input_cursor,
                kill_buffer: String::new(),
                paste_burst: PasteBurst::default(),
                input_history,
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
                vim_enabled: composer_vim_enabled,
                vim_mode: VimMode::Normal,
                vim_pending_d: false,
            },
            viewport: ViewportState::default(),
            goal: GoalState::default(),
            session: SessionState::default(),
            history: Vec::new(),
            history_version: 0,
            history_revisions: Vec::new(),
            next_history_revision: 1,
            api_messages: Vec::new(),
            is_loading: false,
            offline_mode: false,
            turn_error_posted: false,
            status_message: None,
            status_toasts: VecDeque::new(),
            sticky_status: None,
            last_status_message_seen: None,
            model,
            auto_model,
            last_effective_model: None,
            api_provider: provider,
            reasoning_effort,
            last_effective_reasoning_effort: None,
            workspace,
            config_path,
            config_profile,
            mcp_config_path: mcp_config_path.clone(),
            skills_dir,
            memory_path,
            use_memory,
            use_alt_screen,
            use_mouse_capture,
            use_bracketed_paste,
            use_paste_burst_detection,
            bracketed_paste_seen: false,
            system_prompt: None,
            auto_compact,
            calm_mode,
            low_motion,
            fancy_animations,
            show_thinking,
            verbose_transcript: false,
            show_tool_details,
            ui_locale,
            cost_currency,
            composer_density,
            composer_border,
            transcript_spacing,
            sidebar_width_percent,
            sidebar_focus,
            context_panel: settings.context_panel,
            file_tree: None,
            compact_threshold,
            max_input_history,
            allow_shell,
            max_subagents,
            subagent_cache: Vec::new(),
            agent_progress: HashMap::new(),
            subagent_card_index: HashMap::new(),
            last_fanout_card_index: None,
            pending_subagent_dispatch: None,
            agent_activity_started_at: None,
            ui_theme,
            onboarding,
            onboarding_needs_api_key: needs_api_key,
            onboarding_workspace_trust_gate,
            api_key_env_only,
            api_key_input: String::new(),
            api_key_cursor: 0,
            hooks,
            yolo: initial_mode == AppMode::Yolo,
            yolo_restore,
            clipboard: ClipboardHandler::new(),
            approval_session_approved: HashSet::new(),
            approval_session_denied: HashSet::new(),
            approval_mode: if matches!(initial_mode, AppMode::Yolo) {
                ApprovalMode::Auto
            } else {
                config
                    .approval_policy
                    .as_deref()
                    .and_then(ApprovalMode::from_config_value)
                    .unwrap_or_default()
            },
            view_stack: ViewStack::new(),
            backtrack: crate::tui::backtrack::BacktrackState::new(),
            current_session_id: None,
            session_artifacts: Vec::new(),
            trust_mode: initial_mode == AppMode::Yolo,
            status_items: config
                .tui
                .as_ref()
                .and_then(|tui| tui.status_items.clone())
                .unwrap_or_else(crate::config::StatusItem::default_footer),
            project_doc: None,
            plan_state,
            plan_prompt_pending: false,
            plan_tool_used_in_turn: false,
            todos: new_shared_todo_list(),
            runtime_services: RuntimeToolServices {
                shell_manager: Some(shell_manager),
                ..RuntimeToolServices::default()
            },
            mcp_snapshot: None,
            // Read the MCP config once at boot to know how many servers
            // the user has declared. The footer chip uses this even when
            // no live snapshot is available (#502). Cheap (just reads
            // the JSON file); errors fall through to zero so a missing
            // or malformed config simply hides the chip.
            mcp_configured_count: crate::mcp::load_config(&mcp_config_path)
                .map(|cfg| cfg.servers.len())
                .unwrap_or(0),
            mcp_restart_required: false,
            tool_log: Vec::new(),
            active_skill: None,
            cached_skills,
            tool_cells: HashMap::new(),
            tool_details_by_cell: HashMap::new(),
            context_references_by_cell: HashMap::new(),
            session_context_references: Vec::new(),
            active_cell: None,
            active_cell_revision: 0,
            active_tool_details: HashMap::new(),
            exploring_cell: None,
            exploring_entries: HashMap::new(),
            ignored_tool_calls: HashSet::new(),
            last_exec_wait_command: None,
            streaming_message_index: None,
            streaming_thinking_active_entry: None,
            streaming_state: StreamingState::new(),
            reasoning_buffer: String::new(),
            reasoning_header: None,
            last_reasoning: None,
            pending_tool_uses: Vec::new(),
            queued_messages: VecDeque::new(),
            queued_draft: None,
            pending_steers: VecDeque::new(),
            rejected_steers: VecDeque::new(),
            submit_pending_steers_after_interrupt: false,
            turn_started_at: None,
            cumulative_turn_duration: std::time::Duration::ZERO,
            runtime_turn_id: None,
            runtime_turn_status: None,
            dispatch_started_at: None,
            workspace_context: None,
            workspace_context_cell: std::sync::Arc::new(std::sync::Mutex::new(None)),
            workspace_context_refreshed_at: None,
            task_panel: Vec::new(),
            needs_redraw: true,
            thinking_started_at: None,
            is_compacting: false,
            user_scrolled_during_stream: false,
            coherence_state: CoherenceState::default(),
            last_send_at: None,
            quit_armed_until: None,
            cycle_count: 0,
            cycle_briefings: Vec::new(),
            cycle: CycleConfig::default(),
            collapsed_cells: HashSet::new(),
            collapsed_cell_map: Vec::new(),
            edit_in_progress: false,
            lsp_enabled: config.lsp.as_ref().and_then(|l| l.enabled).unwrap_or(true),
            composer_arrows_scroll: config
                .tui
                .as_ref()
                .and_then(|tui| tui.composer_arrows_scroll)
                .unwrap_or(false),
        }
    }

    fn discover_cached_skills(workspace: &std::path::Path) -> Vec<(String, String)> {
        crate::skills::discover_in_workspace(workspace)
            .list()
            .iter()
            .map(|s| (s.name.clone(), s.description.clone()))
            .collect()
    }

    pub fn refresh_skill_cache(&mut self) {
        self.cached_skills = Self::discover_cached_skills(&self.workspace);
    }

    pub fn submit_api_key(&mut self) -> Result<SavedCredential, ApiKeyError> {
        let key = self.api_key_input.trim().to_string();
        if key.is_empty() {
            return Err(ApiKeyError::Empty);
        }

        match save_api_key(&key) {
            Ok(saved) => {
                self.api_key_input.clear();
                self.api_key_cursor = 0;
                self.onboarding_needs_api_key = false;
                self.api_key_env_only = false;
                Ok(saved)
            }
            Err(source) => Err(ApiKeyError::SaveFailed { source }),
        }
    }

    pub fn finish_onboarding(&mut self) {
        self.onboarding = OnboardingState::None;
        if let Err(err) = crate::tui::onboarding::mark_onboarded() {
            self.status_message = Some(format!("Failed to mark onboarding: {err}"));
        }
        self.needs_redraw = true;
    }

    /// Apply a locale tag selected from the onboarding language picker (#566).
    /// Persists the value to `~/.deepseek/settings.toml` and immediately
    /// re-resolves `ui_locale` so the rest of onboarding renders in the new
    /// language. `App` doesn't keep `Settings` resident — it loads on entry
    /// and rewrites on exit, mirroring the pattern used by the `/config`
    /// surface.
    pub fn set_locale_from_onboarding(&mut self, tag: &str) -> anyhow::Result<()> {
        let mut settings = Settings::load().unwrap_or_else(|_| Settings::default());
        settings.set("locale", tag)?;
        settings.save()?;
        self.ui_locale = crate::localization::resolve_locale(&settings.locale);
        self.needs_redraw = true;
        Ok(())
    }

    /// Locale tag currently persisted in `~/.deepseek/settings.toml` (or
    /// `"auto"` when no settings file exists). Used by the onboarding
    /// language picker to highlight the current selection without `App`
    /// having to keep `Settings` resident.
    pub fn current_locale_tag(&self) -> String {
        Settings::load()
            .map(|s| s.locale)
            .unwrap_or_else(|_| "auto".to_string())
    }

    pub fn set_mode(&mut self, mode: AppMode) -> bool {
        let previous_mode = self.mode;
        if previous_mode == mode {
            return false;
        }

        let entering_yolo = mode == AppMode::Yolo && previous_mode != AppMode::Yolo;
        let leaving_yolo = previous_mode == AppMode::Yolo && mode != AppMode::Yolo;
        self.mode = mode;
        self.status_message = Some(format!("Switched to {} mode", mode.label()));

        if entering_yolo {
            self.yolo_restore = Some(YoloRestoreState {
                allow_shell: self.allow_shell,
                trust_mode: self.trust_mode,
                approval_mode: self.approval_mode,
            });
            self.allow_shell = true;
            self.trust_mode = true;
            self.approval_mode = ApprovalMode::Auto;
        } else if leaving_yolo && let Some(restore) = self.yolo_restore.take() {
            self.allow_shell = restore.allow_shell;
            self.trust_mode = restore.trust_mode;
            self.approval_mode = restore.approval_mode;
        }

        self.yolo = mode == AppMode::Yolo;
        if mode != AppMode::Plan {
            self.plan_prompt_pending = false;
            self.plan_tool_used_in_turn = false;
        }

        // Execute mode change hooks
        let context = HookContext::new()
            .with_mode(mode.label())
            .with_previous_mode(previous_mode.label())
            .with_workspace(self.workspace.clone())
            .with_model(&self.model);
        let _ = self.hooks.execute(HookEvent::ModeChange, &context);
        self.needs_redraw = true;
        true
    }

    /// Cycle through modes: Plan → Agent → YOLO → Plan.
    pub fn cycle_mode(&mut self) {
        let next = match self.mode {
            AppMode::Plan => AppMode::Agent,
            AppMode::Agent => AppMode::Yolo,
            AppMode::Yolo => AppMode::Plan,
        };
        let _ = self.set_mode(next);
    }

    /// Cycle through modes in reverse.
    #[allow(dead_code)]
    pub fn cycle_mode_reverse(&mut self) {
        let next = match self.mode {
            AppMode::Agent => AppMode::Plan,
            AppMode::Yolo => AppMode::Agent,
            AppMode::Plan => AppMode::Yolo,
        };
        let _ = self.set_mode(next);
    }

    /// Cycle reasoning-effort through the three behaviorally distinct tiers:
    /// `Off` → `High` → `Max` → `Off`.
    pub fn cycle_effort(&mut self) {
        self.reasoning_effort = self.reasoning_effort.cycle_next();
        self.last_effective_reasoning_effort = None;
        self.needs_redraw = true;
        self.push_status_toast(
            format!("Thinking: {}", self.reasoning_effort.short_label()),
            StatusToastLevel::Info,
            Some(1_500),
        );
    }

    /// Execute hooks for a specific event with the given context
    pub fn execute_hooks(&self, event: HookEvent, context: &HookContext) -> Vec<HookResult> {
        self.hooks.execute(event, context)
    }

    /// Create a hook context with common fields pre-populated
    pub fn base_hook_context(&self) -> HookContext {
        HookContext::new()
            .with_mode(self.mode.label())
            .with_workspace(self.workspace.clone())
            .with_model(&self.model)
            .with_session_id(self.hooks.session_id())
            .with_tokens(self.session.total_tokens)
    }

    /// Soft cap on [`Self::history`] length. When history exceeds this count,
    /// the oldest cells are folded into a single placeholder to bound memory
    /// and render cost (#399 S2). The cap is generous — 5000 cells is more
    /// than enough to keep the visible transcript intact across sessions.
    pub const HISTORY_SOFT_CAP: usize = 5_000;

    /// Number of oldest cells to fold when the soft cap fires. Folding in
    /// batches amortizes the cost instead of triggering on every push.
    const HISTORY_FOLD_BATCH: usize = 1_000;

    pub fn add_message(&mut self, msg: HistoryCell) {
        let rev = self.fresh_history_revision();
        self.history.push(msg);
        self.history_revisions.push(rev);
        self.history_version = self.history_version.wrapping_add(1);

        // Bound history length: when the soft cap fires, fold the oldest
        // batch into a single ArchivedContext placeholder.
        self.maybe_fold_history();
        let selection_has_range = self
            .viewport
            .transcript_selection
            .ordered_endpoints()
            .is_some_and(|(start, end)| start != end);
        if self.viewport.transcript_scroll.is_at_tail()
            && !self.viewport.transcript_selection.dragging
            && !selection_has_range
            && !self.user_scrolled_during_stream
        {
            self.scroll_to_bottom();
        }
    }

    /// Add `delta` to the parent-turn session cost and bump the displayed
    /// high-water mark so the footer total never reverses (#244).
    #[allow(dead_code)]
    pub fn accrue_session_cost(&mut self, delta: f64) {
        self.accrue_session_cost_estimate(CostEstimate::usd_only(delta));
    }

    /// Add a dual-currency parent-turn cost estimate.
    pub fn accrue_session_cost_estimate(&mut self, estimate: CostEstimate) {
        self.session.session_cost += estimate.usd;
        self.session.session_cost_cny += estimate.cny;
        self.refresh_displayed_cost_high_water();
    }

    /// Add `delta` to the running sub-agent cost and bump the displayed
    /// high-water mark so the footer total never reverses (#244).
    #[allow(dead_code)]
    pub fn accrue_subagent_cost(&mut self, delta: f64) {
        self.accrue_subagent_cost_estimate(CostEstimate::usd_only(delta));
    }

    /// Add a dual-currency sub-agent/background cost estimate.
    pub fn accrue_subagent_cost_estimate(&mut self, estimate: CostEstimate) {
        self.session.subagent_cost += estimate.usd;
        self.session.subagent_cost_cny += estimate.cny;
        self.refresh_displayed_cost_high_water();
    }

    /// Recompute the displayed cost high-water mark. Called any time a cost
    /// counter is mutated; never decreases.
    pub fn refresh_displayed_cost_high_water(&mut self) {
        let current = self.session.session_cost + self.session.subagent_cost;
        if current > self.session.displayed_cost_high_water {
            self.session.displayed_cost_high_water = current;
        }
        let current_cny = self.session.session_cost_cny + self.session.subagent_cost_cny;
        if current_cny > self.session.displayed_cost_high_water_cny {
            self.session.displayed_cost_high_water_cny = current_cny;
        }
    }

    /// Read the visible session+sub-agent cost. Guaranteed monotonic across
    /// reconciliation events (cache adjustments, provisional → final swaps)
    /// for the lifetime of one session (#244).
    #[allow(dead_code)]
    pub fn displayed_session_cost(&self) -> f64 {
        self.displayed_session_cost_for_currency(CostCurrency::Usd)
    }

    /// Read the visible session+sub-agent cost in the chosen currency.
    pub fn displayed_session_cost_for_currency(&self, currency: CostCurrency) -> f64 {
        match currency {
            CostCurrency::Usd => {
                let current = self.session.session_cost + self.session.subagent_cost;
                current.max(self.session.displayed_cost_high_water)
            }
            CostCurrency::Cny => {
                let current = self.session.session_cost_cny + self.session.subagent_cost_cny;
                current.max(self.session.displayed_cost_high_water_cny)
            }
        }
    }

    pub fn session_cost_for_currency(&self, currency: CostCurrency) -> f64 {
        match currency {
            CostCurrency::Usd => self.session.session_cost,
            CostCurrency::Cny => self.session.session_cost_cny,
        }
    }

    pub fn subagent_cost_for_currency(&self, currency: CostCurrency) -> f64 {
        match currency {
            CostCurrency::Usd => self.session.subagent_cost,
            CostCurrency::Cny => self.session.subagent_cost_cny,
        }
    }

    pub fn format_cost_amount(&self, amount: f64) -> String {
        crate::pricing::format_cost_amount(amount, self.cost_currency)
    }

    pub fn format_cost_amount_precise(&self, amount: f64) -> String {
        crate::pricing::format_cost_amount_precise(amount, self.cost_currency)
    }

    /// Fold the oldest [`Self::HISTORY_FOLD_BATCH`] cells into a single
    /// `ArchivedContext` placeholder when history exceeds the soft cap.
    /// Called from [`Self::add_message`]; the caller is responsible for
    /// also removing the folded range from any auxiliary per-cell maps.
    fn maybe_fold_history(&mut self) {
        if self.history.len() <= Self::HISTORY_SOFT_CAP {
            return;
        }

        let fold_count = Self::HISTORY_FOLD_BATCH.min(self.history.len());
        // Don't fold into the very last cell(s) — keep a buffer of
        // non-folded cells so the visible transcript tail stays intact.
        let keep_tail = Self::HISTORY_SOFT_CAP.saturating_sub(Self::HISTORY_FOLD_BATCH);
        if self.history.len().saturating_sub(fold_count) < keep_tail {
            return;
        }

        // Gather the range of cell indices we are folding.
        let folded: Vec<HistoryCell> = self.history.drain(..fold_count).collect();
        let folded_revs: Vec<u64> = self.history_revisions.drain(..fold_count).collect();
        let _ = folded_revs; // revisions are discarded with the cells

        // Shift all per-cell index maps down by `fold_count`.
        self.shift_history_maps_down(fold_count);

        // Build a single placeholder cell summarizing the folded range.
        let total_folded = folded.len();
        let summary = format!(
            "{total_folded} older transcript cells folded to bound memory. \
             Use /sessions to load a prior session snapshot if needed."
        );
        let placeholder = HistoryCell::ArchivedContext {
            level: 0,
            range: format!("cells 0-{}", total_folded.saturating_sub(1)),
            tokens: String::new(),
            density: String::new(),
            model: String::new(),
            timestamp: String::new(),
            summary,
        };

        // Insert the placeholder at the front.
        let rev = self.fresh_history_revision();
        self.history.insert(0, placeholder);
        self.history_revisions.insert(0, rev);
        self.history_version = self.history_version.wrapping_add(1);
        self.needs_redraw = true;
    }

    /// Shift all per-cell index maps down by `n` after removing the first
    /// `n` history cells. Every map key >= n is mapped to key - n; keys < n
    /// are dropped.
    fn shift_history_maps_down(&mut self, n: usize) {
        // tool_cells: HashMap<String, usize>
        self.tool_cells.retain(|_, idx| {
            if *idx >= n {
                *idx -= n;
                true
            } else {
                false
            }
        });

        // tool_details_by_cell: HashMap<usize, ToolDetailRecord>
        self.tool_details_by_cell = std::mem::take(&mut self.tool_details_by_cell)
            .into_iter()
            .filter_map(|(idx, detail)| {
                if idx >= n {
                    Some((idx - n, detail))
                } else {
                    None
                }
            })
            .collect();

        // context_references_by_cell
        self.context_references_by_cell = std::mem::take(&mut self.context_references_by_cell)
            .into_iter()
            .filter_map(|(idx, refs)| {
                if idx >= n {
                    Some((idx - n, refs))
                } else {
                    None
                }
            })
            .collect();
        self.rebuild_session_context_references();

        // subagent_card_index
        self.subagent_card_index.retain(|_, idx| {
            if *idx >= n {
                *idx -= n;
                true
            } else {
                false
            }
        });

        // last_fanout_card_index
        if let Some(ref mut idx) = self.last_fanout_card_index {
            if *idx >= n {
                *idx -= n;
            } else {
                self.last_fanout_card_index = None;
            }
        }

        // collapsed_cells
        self.collapsed_cells = std::mem::take(&mut self.collapsed_cells)
            .into_iter()
            .filter_map(|idx| if idx >= n { Some(idx - n) } else { None })
            .collect();
        self.collapsed_cell_map.clear();
    }

    pub fn mark_history_updated(&mut self) {
        self.history_version = self.history_version.wrapping_add(1);
        // Resync per-cell revisions to history.len(). This is the
        // "I-don't-know-which-cell-changed" path: if cells were appended in
        // bulk (e.g. session resume, compaction), every new cell gets a
        // fresh revision; if cells were removed, drop trailing revs. We
        // intentionally do NOT bump revisions for indices that already had
        // one — the cache will reuse those. Callers that mutate a specific
        // cell's content must call `bump_history_cell(idx)` instead.
        self.resync_history_revisions();
        self.needs_redraw = true;
    }

    /// Issue a fresh, monotonically increasing revision counter for a new
    /// history cell. Wrapping is acceptable — collisions are astronomically
    /// rare and at worst trigger one extra re-render.
    fn fresh_history_revision(&mut self) -> u64 {
        let rev = self.next_history_revision;
        self.next_history_revision = self.next_history_revision.wrapping_add(1);
        rev
    }

    /// Bring `history_revisions` back into shape (`history_revisions.len() ==
    /// history.len()`). Pushes fresh revs for newly appended cells, truncates
    /// for cells that were removed. **Does not** invalidate existing entries.
    pub fn resync_history_revisions(&mut self) {
        if self.history_revisions.len() < self.history.len() {
            let needed = self.history.len() - self.history_revisions.len();
            for _ in 0..needed {
                let rev = self.fresh_history_revision();
                self.history_revisions.push(rev);
            }
        } else if self.history_revisions.len() > self.history.len() {
            self.history_revisions.truncate(self.history.len());
        }
    }

    /// Bump the revision counter of a single history cell so the transcript
    /// cache re-renders it on the next frame. Use this whenever a cell's
    /// content (e.g. a streaming Assistant body) is mutated in place.
    pub fn bump_history_cell(&mut self, idx: usize) {
        // Resync first in case callers mutated `history` directly without
        // pushing through `add_message`. After resync, the index is valid
        // (or out of bounds — in which case there's nothing to bump).
        self.resync_history_revisions();
        if let Some(rev) = self.history_revisions.get_mut(idx) {
            let new_rev = self.next_history_revision;
            self.next_history_revision = self.next_history_revision.wrapping_add(1);
            *rev = new_rev;
        }
        self.history_version = self.history_version.wrapping_add(1);
        self.needs_redraw = true;
    }

    /// Append a single history cell, allocating a fresh per-cell revision.
    /// Equivalent to `add_message` but exposed as a generic alias so call
    /// sites currently doing `app.history.push(...)` followed by
    /// `app.mark_history_updated()` can collapse to one helper.
    pub fn push_history_cell(&mut self, cell: HistoryCell) {
        let rev = self.fresh_history_revision();
        self.history.push(cell);
        self.history_revisions.push(rev);
        self.history_version = self.history_version.wrapping_add(1);
        self.maybe_fold_history();
        self.needs_redraw = true;
    }

    /// Append a batch of history cells, allocating fresh revisions.
    pub fn extend_history<I>(&mut self, cells: I)
    where
        I: IntoIterator<Item = HistoryCell>,
    {
        for cell in cells {
            let rev = self.fresh_history_revision();
            self.history.push(cell);
            self.history_revisions.push(rev);
        }
        self.maybe_fold_history();
        self.history_version = self.history_version.wrapping_add(1);
        self.needs_redraw = true;
    }

    /// Clear the history and its session-scoped side indexes. Used by /clear,
    /// session reset, and other "wipe and reload" flows.
    pub fn clear_history(&mut self) {
        self.history.clear();
        self.history_revisions.clear();
        self.context_references_by_cell.clear();
        self.session_context_references.clear();
        self.session_artifacts.clear();
        self.collapsed_cells.clear();
        self.collapsed_cell_map.clear();
        self.history_version = self.history_version.wrapping_add(1);
        self.needs_redraw = true;
    }

    /// Pop the trailing history cell, keeping revisions in sync.
    pub fn pop_history(&mut self) -> Option<HistoryCell> {
        let cell = self.history.pop();
        if cell.is_some() {
            self.history_revisions.pop();
            self.context_references_by_cell.remove(&self.history.len());
            self.rebuild_session_context_references();
            self.history_version = self.history_version.wrapping_add(1);
            self.needs_redraw = true;
        }
        cell
    }

    /// Truncate `history` (and the parallel `history_revisions` + auxiliary
    /// per-cell maps) so that only cells with index `< new_len` remain.
    /// Used by Esc-Esc backtrack (#133) to roll the visible transcript
    /// back to a chosen user message. Cells dropped here are gone — the
    /// caller is expected to also trim the matching `api_messages` so the
    /// next turn matches what the user sees.
    pub fn truncate_history_to(&mut self, new_len: usize) {
        if new_len >= self.history.len() {
            return;
        }
        self.history.truncate(new_len);
        if self.history_revisions.len() > new_len {
            self.history_revisions.truncate(new_len);
        }
        // Drop any auxiliary maps keyed on history indices that now point
        // past the new tail. We keep the rest intact so unaffected tool
        // cells continue to render correctly.
        self.tool_cells.retain(|_, idx| *idx < new_len);
        self.tool_details_by_cell.retain(|idx, _| *idx < new_len);
        self.context_references_by_cell
            .retain(|idx, _| *idx < new_len);
        self.rebuild_session_context_references();
        self.subagent_card_index.retain(|_, idx| *idx < new_len);
        if self
            .last_fanout_card_index
            .is_some_and(|idx| idx >= new_len)
        {
            self.last_fanout_card_index = None;
        }
        // Drop collapsed cells that reference indices past the new tail.
        self.collapsed_cells.retain(|idx| *idx < new_len);
        self.collapsed_cell_map.clear();
        self.history_version = self.history_version.wrapping_add(1);
        self.needs_redraw = true;
    }

    /// Bump the active-cell revision counter and request a redraw.
    ///
    /// Use this whenever an entry inside `active_cell` is mutated. The
    /// transcript cache combines this counter with `history_version` to
    /// produce a per-cell revision so the synthetic active-cell row can be
    /// re-rendered without invalidating committed history cells.
    pub fn bump_active_cell_revision(&mut self) {
        self.active_cell_revision = self.active_cell_revision.wrapping_add(1);
        if let Some(active) = self.active_cell.as_mut() {
            active.bump_revision();
        }
        self.history_version = self.history_version.wrapping_add(1);
        self.needs_redraw = true;
    }

    /// Total number of cells in the *virtual* transcript: `history.len()`
    /// plus active cell entries (if any).
    #[must_use]
    #[allow(dead_code)] // Reserved for renderers that need a unified cell count.
    pub fn virtual_cell_count(&self) -> usize {
        self.history.len() + self.active_cell.as_ref().map_or(0, ActiveCell::entry_count)
    }

    /// The next cell index a freshly-pushed entry would occupy in the virtual
    /// transcript. Used by `register_tool_cell`-style callsites that record
    /// cell-index metadata before the active cell flushes to history.
    #[must_use]
    #[allow(dead_code)] // Reserved for the eventual merged push helper.
    pub fn next_virtual_cell_index(&self) -> usize {
        self.virtual_cell_count()
    }

    /// Resolve a virtual cell index to either a committed history cell or an
    /// active-cell entry. Used by the pager / details lookup code so it can
    /// transparently address still-in-flight cells.
    #[must_use]
    #[allow(dead_code)] // Used by the upcoming pager rewrite (read-only resolver).
    pub fn cell_at_virtual_index(&self, index: usize) -> Option<&HistoryCell> {
        if index < self.history.len() {
            self.history.get(index)
        } else {
            let entry_idx = index - self.history.len();
            self.active_cell
                .as_ref()
                .and_then(|active| active.entries().get(entry_idx))
        }
    }

    /// Resolve the tool-detail record for a committed or still-active virtual
    /// transcript cell.
    #[must_use]
    pub fn tool_detail_record_for_cell(&self, index: usize) -> Option<&ToolDetailRecord> {
        if let Some(detail) = self.tool_details_by_cell.get(&index) {
            return Some(detail);
        }
        self.active_tool_details
            .values()
            .find(|detail| self.tool_cells.get(&detail.tool_id).copied() == Some(index))
    }

    /// Whether a virtual transcript cell can open a meaningful Alt+V detail
    /// view.
    #[must_use]
    pub fn cell_has_detail_target(&self, index: usize) -> bool {
        self.tool_detail_record_for_cell(index).is_some()
            || matches!(
                self.cell_at_virtual_index(index),
                Some(HistoryCell::Tool(_) | HistoryCell::SubAgent(_))
            )
    }

    /// Pick the detail target for the current viewport. This is used by the
    /// transcript highlight and footer hint so they agree with Alt+V.
    #[must_use]
    pub fn detail_cell_index_for_viewport(
        &self,
        top: usize,
        visible: usize,
        line_meta: &[TranscriptLineMeta],
    ) -> Option<usize> {
        let selected_cell = self
            .viewport
            .transcript_selection
            .ordered_endpoints()
            .and_then(|(start, _)| line_meta.get(start.line_index))
            .and_then(TranscriptLineMeta::cell_line)
            .map(|(cell_index, _)| cell_index)
            .filter(|&idx| self.cell_has_detail_target(idx));
        if selected_cell.is_some() {
            return selected_cell;
        }

        let start = top.min(line_meta.len().saturating_sub(1));
        let end = start.saturating_add(visible).min(line_meta.len());
        for meta in line_meta.iter().take(end).skip(start) {
            let Some((cell_index, _)) = meta.cell_line() else {
                continue;
            };
            if self.cell_has_detail_target(cell_index) {
                return Some(cell_index);
            }
        }

        (0..self.virtual_cell_count())
            .rev()
            .find(|&idx| self.cell_has_detail_target(idx))
    }

    pub fn record_context_references(
        &mut self,
        history_cell: usize,
        message_index: usize,
        references: Vec<ContextReference>,
    ) {
        if references.is_empty() {
            return;
        }
        let records: Vec<SessionContextReference> = references
            .into_iter()
            .map(|reference| SessionContextReference {
                message_index,
                reference,
            })
            .collect();
        self.context_references_by_cell
            .insert(history_cell, records.clone());
        self.rebuild_session_context_references();
        self.needs_redraw = true;
    }

    pub fn sync_context_references_from_session(
        &mut self,
        references: &[SessionContextReference],
        message_to_cell: &HashMap<usize, usize>,
    ) {
        self.context_references_by_cell.clear();
        for record in references {
            let Some(&cell_index) = message_to_cell.get(&record.message_index) else {
                continue;
            };
            self.context_references_by_cell
                .entry(cell_index)
                .or_default()
                .push(record.clone());
        }
        self.rebuild_session_context_references();
    }

    fn rebuild_session_context_references(&mut self) {
        let mut records: Vec<SessionContextReference> = self
            .context_references_by_cell
            .values()
            .flat_map(|records| records.iter().cloned())
            .collect();
        records.sort_by_key(|record| record.message_index);
        self.session_context_references = records;
    }

    /// Mutable variant of [`Self::cell_at_virtual_index`]. Bumps the
    /// appropriate revision counter (active-cell revision when targeting an
    /// in-flight entry, history version otherwise).
    pub fn cell_at_virtual_index_mut(&mut self, index: usize) -> Option<&mut HistoryCell> {
        if index < self.history.len() {
            // Bump only the targeted cell's revision; leave every other
            // cell's cached render intact.
            self.resync_history_revisions();
            if let Some(rev) = self.history_revisions.get_mut(index) {
                let new_rev = self.next_history_revision;
                self.next_history_revision = self.next_history_revision.wrapping_add(1);
                *rev = new_rev;
            }
            self.history_version = self.history_version.wrapping_add(1);
            self.history.get_mut(index)
        } else {
            let entry_idx = index - self.history.len();
            self.active_cell_revision = self.active_cell_revision.wrapping_add(1);
            self.history_version = self.history_version.wrapping_add(1);
            self.active_cell
                .as_mut()
                .and_then(|active| active.entry_mut(entry_idx))
        }
    }

    /// Drain the active cell into history. Companion maps that reference
    /// active-cell entries by virtual index (`tool_cells`,
    /// `tool_details_by_cell`) are rewritten to point at the new history
    /// indices. Idempotent — calling this when there is no active cell is a
    /// no-op.
    ///
    /// Caller is responsible for first marking in-progress entries with the
    /// terminal status they want (e.g. via
    /// [`ActiveCell::mark_in_progress_as_interrupted`]).
    pub fn flush_active_cell(&mut self) {
        let Some(mut active) = self.active_cell.take() else {
            self.streaming_thinking_active_entry = None;
            return;
        };
        if active.is_empty() {
            self.exploring_cell = None;
            self.exploring_entries.clear();
            self.active_tool_details.clear();
            self.streaming_thinking_active_entry = None;
            self.bump_active_cell_revision();
            return;
        }

        if let Some(entry_idx) = self.streaming_thinking_active_entry.take()
            && let Some(HistoryCell::Thinking { streaming, .. }) = active.entry_mut(entry_idx)
        {
            *streaming = false;
        }

        let drained = active.drain();
        let base_index = self.history.len();

        let mut details = std::mem::take(&mut self.active_tool_details);
        for (tool_id, detail) in details.drain() {
            self.tool_details_by_cell
                .entry(self.tool_cells.get(&tool_id).copied().unwrap_or(base_index))
                .or_insert(detail);
        }

        self.exploring_cell = None;
        self.exploring_entries.clear();

        for cell in drained {
            let rev = self.fresh_history_revision();
            self.history.push(cell);
            self.history_revisions.push(rev);
        }
        self.history_version = self.history_version.wrapping_add(1);
        self.needs_redraw = true;
        let selection_has_range = self
            .viewport
            .transcript_selection
            .ordered_endpoints()
            .is_some_and(|(start, end)| start != end);
        if self.viewport.transcript_scroll.is_at_tail()
            && !self.viewport.transcript_selection.dragging
            && !selection_has_range
            && !self.user_scrolled_during_stream
        {
            self.scroll_to_bottom();
        }
    }

    /// Mark every still-running entry in the active cell as interrupted, then
    /// flush. Convenience helper for cancellation paths.
    pub fn finalize_active_cell_as_interrupted(&mut self) {
        if let Some(active) = self.active_cell.as_mut() {
            active.mark_in_progress_as_interrupted();
        }
        self.flush_active_cell();
    }

    pub fn push_status_toast(
        &mut self,
        text: impl Into<String>,
        level: StatusToastLevel,
        ttl_ms: Option<u64>,
    ) {
        let toast = StatusToast::new(text, level, ttl_ms);
        self.status_toasts.push_back(toast);
        while self.status_toasts.len() > 24 {
            self.status_toasts.pop_front();
        }
        self.needs_redraw = true;
    }

    /// How long the "press Ctrl+C again to quit" prompt stays armed before it
    /// silently expires.
    pub const QUIT_CONFIRMATION_WINDOW: Duration = Duration::from_secs(2);

    /// Arm the quit confirmation timer. The next Ctrl+C within
    /// [`Self::QUIT_CONFIRMATION_WINDOW`] should exit the app cleanly. Call this only
    /// from idle state — while a turn is in flight or a modal is open Ctrl+C
    /// retains its existing "interrupt this turn" / "close modal" semantics.
    pub fn arm_quit(&mut self) {
        self.quit_armed_until = Some(Instant::now() + Self::QUIT_CONFIRMATION_WINDOW);
        self.needs_redraw = true;
    }

    /// Whether the quit timer is currently armed (i.e. a prior Ctrl+C set it
    /// and it hasn't expired yet).
    pub fn quit_is_armed(&self) -> bool {
        self.quit_armed_until
            .map(|deadline| Instant::now() < deadline)
            .unwrap_or(false)
    }

    /// Clear the quit-armed timer. Call when expiry is detected on a tick or
    /// when the user takes any other action that should disarm the prompt
    /// (typing, sending a message, etc.).
    pub fn disarm_quit(&mut self) {
        if self.quit_armed_until.is_some() {
            self.quit_armed_until = None;
            self.needs_redraw = true;
        }
    }

    /// Tick called from the redraw loop. Lets time-based UI state (the
    /// quit-armed prompt) expire even when no input event is delivered.
    pub fn tick_quit_armed(&mut self) {
        if let Some(deadline) = self.quit_armed_until
            && Instant::now() >= deadline
        {
            self.quit_armed_until = None;
            self.needs_redraw = true;
        }
    }

    pub fn set_sticky_status(
        &mut self,
        text: impl Into<String>,
        level: StatusToastLevel,
        ttl_ms: Option<u64>,
    ) {
        self.sticky_status = Some(StatusToast::new(text, level, ttl_ms));
        self.needs_redraw = true;
    }

    pub fn clear_sticky_status(&mut self) {
        self.sticky_status = None;
    }

    pub fn set_sidebar_focus(&mut self, focus: SidebarFocus) {
        self.sidebar_focus = focus;
        self.needs_redraw = true;
    }

    pub fn close_slash_menu(&mut self) {
        self.slash_menu_hidden = true;
        self.needs_redraw = true;
    }

    fn classify_status_text(text: &str) -> (StatusToastLevel, Option<u64>, bool) {
        let lower = text.to_ascii_lowercase();
        let has = |needle: &str| lower.contains(needle);

        if has("offline mode") || has("context critical") {
            return (StatusToastLevel::Warning, None, true);
        }
        if has("error")
            || has("failed")
            || has("denied")
            || has("timeout")
            || has("aborted")
            || has("critical")
        {
            return (StatusToastLevel::Error, Some(15_000), true);
        }
        if has("saved")
            || has("loaded")
            || has("queued")
            || has("found")
            || has("enabled")
            || has("completed")
        {
            return (StatusToastLevel::Success, Some(5_000), false);
        }
        if has("cancelled") || has("warning") {
            return (StatusToastLevel::Warning, Some(5_000), false);
        }
        (StatusToastLevel::Info, Some(4_000), false)
    }

    pub fn sync_status_message_to_toasts(&mut self) {
        let current = self.status_message.clone();
        if self.last_status_message_seen == current {
            return;
        }
        self.last_status_message_seen = current.clone();

        let Some(message) = current else {
            return;
        };
        if message.trim().is_empty() {
            return;
        }

        let (level, ttl_ms, sticky) = Self::classify_status_text(&message);
        if sticky {
            self.set_sticky_status(message, level, ttl_ms);
        } else {
            if matches!(level, StatusToastLevel::Success)
                && self
                    .sticky_status
                    .as_ref()
                    .is_some_and(|toast| matches!(toast.level, StatusToastLevel::Error))
            {
                self.clear_sticky_status();
            }
            self.push_status_toast(message, level, ttl_ms);
        }
    }

    /// Up to `limit` currently-active toasts, most recent last (so a stacked
    /// renderer iterating top-to-bottom shows the freshest message at the
    /// bottom, like a chat log). Drains expired toasts off the front as a
    /// side effect — same cleanup as `active_status_toast` so callers see a
    /// consistent queue. Whalescale#439.
    pub fn active_status_toasts(&mut self, limit: usize) -> Vec<StatusToast> {
        self.sync_status_message_to_toasts();
        let now = Instant::now();
        while self
            .status_toasts
            .front()
            .is_some_and(|toast| toast.is_expired(now))
        {
            self.status_toasts.pop_front();
            self.needs_redraw = true;
        }
        if self
            .sticky_status
            .as_ref()
            .is_some_and(|toast| toast.is_expired(now))
        {
            self.sticky_status = None;
            self.needs_redraw = true;
        }

        let mut out: Vec<StatusToast> = Vec::with_capacity(limit);
        if let Some(sticky) = self.sticky_status.clone() {
            out.push(sticky);
        }
        let take = limit.saturating_sub(out.len());
        let queued: Vec<StatusToast> = self
            .status_toasts
            .iter()
            .rev()
            .take(take)
            .cloned()
            .collect();
        // Iterate in queue order (oldest of the visible window first) so the
        // stacked renderer feels chronological — most recent at the bottom.
        for toast in queued.into_iter().rev() {
            out.push(toast);
        }
        out
    }

    pub fn active_status_toast(&mut self) -> Option<StatusToast> {
        self.sync_status_message_to_toasts();
        let now = Instant::now();
        let mut removed = false;

        while self
            .status_toasts
            .front()
            .is_some_and(|toast| toast.is_expired(now))
        {
            self.status_toasts.pop_front();
            removed = true;
        }

        if self
            .sticky_status
            .as_ref()
            .is_some_and(|toast| toast.is_expired(now))
        {
            self.sticky_status = None;
            removed = true;
        }

        if removed {
            self.needs_redraw = true;
        }

        self.sticky_status
            .clone()
            .or_else(|| self.status_toasts.back().cloned())
    }

    pub fn transcript_render_options(&self) -> TranscriptRenderOptions {
        TranscriptRenderOptions {
            show_thinking: self.show_thinking,
            verbose: self.verbose_transcript,
            show_tool_details: self.show_tool_details,
            calm_mode: self.calm_mode,
            low_motion: self.low_motion,
            spacing: self.transcript_spacing,
        }
    }

    /// Handle terminal resize event.
    pub fn handle_resize(&mut self, _width: u16, _height: u16) {
        self.viewport.transcript_cache = TranscriptViewCache::new();

        if !self.viewport.transcript_scroll.is_at_tail() {
            self.viewport.transcript_scroll = TranscriptScroll::to_bottom();
        }

        self.viewport.pending_scroll_delta = 0;
        self.viewport.transcript_selection.clear();

        self.viewport.last_transcript_area = None;
        self.viewport.last_transcript_top = 0;
        self.viewport.last_transcript_visible = 0;
        self.viewport.last_transcript_total = 0;
        self.viewport.last_transcript_padding_top = 0;
        self.viewport.jump_to_latest_button_area = None;

        self.mark_history_updated();
    }

    pub fn cursor_byte_index(&self) -> usize {
        byte_index_at_char(&self.input, self.cursor_position)
    }

    pub fn insert_str(&mut self, text: &str) {
        if text.is_empty() {
            return;
        }
        self.selected_attachment_index = None;
        let cursor = self.cursor_position.min(char_count(&self.input));
        let byte_index = byte_index_at_char(&self.input, cursor);
        self.input.insert_str(byte_index, text);
        self.cursor_position = cursor + char_count(text);
        self.slash_menu_hidden = false;
        self.mention_menu_hidden = false;
        self.mention_menu_selected = 0;
        self.needs_redraw = true;
    }

    pub fn insert_paste_text(&mut self, text: &str) {
        if let Some(pending) = self.paste_burst.flush_before_modified_input() {
            self.insert_str(&pending);
        }
        let normalized = normalize_paste_text(text);
        if !normalized.is_empty() {
            self.insert_str(&normalized);
        }
        self.paste_burst.clear_after_explicit_paste();
        // Visible-before-submit consolidation: when the post-paste input
        // is over the cap, swap it for an @paste-…md mention immediately
        // (instead of waiting until the user presses Enter and getting
        // surprised by an auto-sent @mention). The same logic runs as a
        // safety-net at submit time so any other code path that fills
        // self.input above the cap still consolidates rather than
        // silently truncating.
        self.consolidate_large_input_if_oversized();
    }

    pub fn insert_media_attachment(&mut self, kind: &str, path: &Path, description: Option<&str>) {
        let reference = media_attachment_reference(kind, path, description);
        let cursor = self.cursor_position.min(char_count(&self.input));
        let byte_index = byte_index_at_char(&self.input, cursor);
        let needs_prefix_newline = self.input[..byte_index]
            .chars()
            .last()
            .is_some_and(|ch| !ch.is_whitespace());
        let needs_suffix_newline = self.input[byte_index..]
            .chars()
            .next()
            .is_some_and(|ch| !ch.is_whitespace());

        let mut inserted = String::new();
        if needs_prefix_newline {
            inserted.push('\n');
        }
        inserted.push_str(&reference);
        if needs_suffix_newline || self.input[byte_index..].is_empty() {
            inserted.push('\n');
        }
        self.insert_str(&inserted);
        self.paste_burst.clear_after_explicit_paste();
    }

    pub fn composer_attachment_count(&self) -> usize {
        crate::tui::file_mention::media_attachment_references(&self.input).len()
    }

    pub fn selected_composer_attachment_index(&self) -> Option<usize> {
        let count = self.composer_attachment_count();
        self.selected_attachment_index
            .filter(|index| *index < count)
    }

    pub fn select_previous_composer_attachment(&mut self) -> bool {
        let count = self.composer_attachment_count();
        if count == 0 {
            self.selected_attachment_index = None;
            return false;
        }

        let next = self
            .selected_composer_attachment_index()
            .map_or(count.saturating_sub(1), |index| index.saturating_sub(1));
        self.selected_attachment_index = Some(next);
        self.cursor_position = 0;
        self.status_message = Some("Attachment selected - Backspace/Delete removes it".to_string());
        self.needs_redraw = true;
        true
    }

    pub fn select_next_composer_attachment(&mut self) -> bool {
        let count = self.composer_attachment_count();
        let Some(index) = self.selected_composer_attachment_index() else {
            return false;
        };
        if index + 1 < count {
            self.selected_attachment_index = Some(index + 1);
            self.status_message =
                Some("Attachment selected - Backspace/Delete removes it".to_string());
        } else {
            self.selected_attachment_index = None;
            self.status_message = Some("Composer focused".to_string());
        }
        self.needs_redraw = true;
        true
    }

    pub fn clear_composer_attachment_selection(&mut self) -> bool {
        if self.selected_attachment_index.take().is_some() {
            self.status_message = Some("Composer focused".to_string());
            self.needs_redraw = true;
            true
        } else {
            false
        }
    }

    pub fn remove_selected_composer_attachment(&mut self) -> bool {
        let references = crate::tui::file_mention::media_attachment_references(&self.input);
        let Some(index) = self
            .selected_composer_attachment_index()
            .filter(|index| *index < references.len())
        else {
            self.selected_attachment_index = None;
            return false;
        };
        let reference = references[index].clone();
        let cursor_byte = byte_index_at_char(&self.input, self.cursor_position);
        let new_cursor_byte = if cursor_byte <= reference.start_byte {
            cursor_byte
        } else if cursor_byte >= reference.end_byte {
            cursor_byte.saturating_sub(reference.end_byte - reference.start_byte)
        } else {
            reference.start_byte
        };

        self.input
            .replace_range(reference.start_byte..reference.end_byte, "");
        self.cursor_position = self.input[..new_cursor_byte.min(self.input.len())]
            .chars()
            .count();
        let remaining = self.composer_attachment_count();
        self.selected_attachment_index = if remaining == 0 {
            None
        } else {
            Some(index.min(remaining.saturating_sub(1)))
        };
        self.slash_menu_hidden = false;
        self.mention_menu_hidden = false;
        self.mention_menu_selected = 0;
        self.status_message = Some(format!("Removed attachment: {}", reference.path));
        self.needs_redraw = true;
        true
    }

    pub fn flush_paste_burst_if_due(&mut self, now: Instant) -> bool {
        match self.paste_burst.flush_if_due(now) {
            FlushResult::Paste(text) => {
                self.insert_str(&text);
                true
            }
            FlushResult::Typed(ch) => {
                self.insert_char(ch);
                true
            }
            FlushResult::None => false,
        }
    }

    pub fn flush_paste_burst_if_enabled(&mut self, now: Instant) -> bool {
        self.use_paste_burst_detection && self.flush_paste_burst_if_due(now)
    }

    pub fn paste_burst_next_flush_delay_if_enabled(&self, now: Instant) -> Option<Duration> {
        if self.use_paste_burst_detection {
            self.paste_burst.next_flush_delay(now)
        } else {
            None
        }
    }

    pub fn flush_paste_burst_before_modified_input_if_enabled(&mut self) -> Option<String> {
        if self.use_paste_burst_detection {
            self.paste_burst.flush_before_modified_input()
        } else {
            None
        }
    }

    pub fn insert_api_key_char(&mut self, c: char) {
        let cursor = self.api_key_cursor.min(char_count(&self.api_key_input));
        let byte_index = byte_index_at_char(&self.api_key_input, cursor);
        self.api_key_input.insert(byte_index, c);
        self.api_key_cursor = cursor + 1;
    }

    pub fn insert_api_key_str(&mut self, text: &str) {
        let sanitized = sanitize_api_key_text(text);
        if sanitized.is_empty() {
            return;
        }
        let cursor = self.api_key_cursor.min(char_count(&self.api_key_input));
        let byte_index = byte_index_at_char(&self.api_key_input, cursor);
        self.api_key_input.insert_str(byte_index, &sanitized);
        self.api_key_cursor = cursor + char_count(&sanitized);
    }

    pub fn delete_api_key_char(&mut self) {
        if self.api_key_cursor == 0 {
            return;
        }
        let target = self.api_key_cursor.saturating_sub(1);
        if remove_char_at(&mut self.api_key_input, target) {
            self.api_key_cursor = target;
        }
    }

    /// Paste from clipboard into input
    pub fn paste_from_clipboard(&mut self) {
        if let Some(content) = self.clipboard.read(self.workspace.as_path()) {
            self.apply_clipboard_content(content);
        }
    }

    pub fn apply_clipboard_content(&mut self, content: ClipboardContent) {
        match content {
            ClipboardContent::Text(text) => {
                self.insert_paste_text(&text);
            }
            ClipboardContent::Image(pasted) => {
                let description = format!("{} ({})", pasted.short_label(), pasted.size_label());
                self.insert_media_attachment("image", &pasted.path, Some(&description));
                self.status_message = Some(format!("Attached image: {description}"));
            }
        }
    }

    pub fn paste_api_key_from_clipboard(&mut self) {
        if let Some(ClipboardContent::Text(text)) = self.clipboard.read(self.workspace.as_path()) {
            self.insert_api_key_str(&text);
        }
    }

    pub fn scroll_up(&mut self, amount: usize) {
        let delta = i32::try_from(amount).unwrap_or(i32::MAX);
        self.viewport.pending_scroll_delta =
            self.viewport.pending_scroll_delta.saturating_sub(delta);
        self.user_scrolled_during_stream = true;
        self.needs_redraw = true;
    }

    pub fn scroll_down(&mut self, amount: usize) {
        let delta = i32::try_from(amount).unwrap_or(i32::MAX);
        self.viewport.pending_scroll_delta =
            self.viewport.pending_scroll_delta.saturating_add(delta);
        self.user_scrolled_during_stream = true;
        self.needs_redraw = true;
    }

    pub fn scroll_to_bottom(&mut self) {
        self.viewport.transcript_scroll = TranscriptScroll::to_bottom();
        self.viewport.pending_scroll_delta = 0;
        self.viewport.jump_to_latest_button_area = None;
        self.user_scrolled_during_stream = false;
        self.needs_redraw = true;
    }

    pub fn insert_char(&mut self, c: char) {
        self.clear_input_history_navigation();
        self.selected_attachment_index = None;
        let cursor = self.cursor_position.min(char_count(&self.input));
        let byte_index = byte_index_at_char(&self.input, cursor);
        self.input.insert(byte_index, c);
        self.cursor_position = cursor + 1;
        self.slash_menu_hidden = false;
        self.mention_menu_hidden = false;
        self.mention_menu_selected = 0;
        self.needs_redraw = true;
    }

    pub fn delete_char(&mut self) {
        self.clear_input_history_navigation();
        self.selected_attachment_index = None;
        if self.cursor_position == 0 {
            return;
        }
        let target = self.cursor_position.saturating_sub(1);
        let removed = remove_char_at(&mut self.input, target);
        if removed {
            self.cursor_position = target;
            self.slash_menu_hidden = false;
            self.mention_menu_hidden = false;
            self.mention_menu_selected = 0;
            self.needs_redraw = true;
        }
    }

    pub fn delete_char_forward(&mut self) {
        self.clear_input_history_navigation();
        self.selected_attachment_index = None;
        if self.input.is_empty() {
            return;
        }
        let target = self.cursor_position;
        let removed = remove_char_at(&mut self.input, target);
        if !removed {
            self.cursor_position = char_count(&self.input);
        }
        self.slash_menu_hidden = false;
        self.mention_menu_hidden = false;
        self.mention_menu_selected = 0;
        self.needs_redraw = true;
    }

    /// Delete the word before the cursor.
    pub fn delete_word_backward(&mut self) {
        self.clear_input_history_navigation();
        self.selected_attachment_index = None;
        if self.cursor_position == 0 {
            return;
        }

        let cursor_byte = byte_index_at_char(&self.input, self.cursor_position);
        let mut word_start = cursor_byte;

        while word_start > 0 {
            let Some((prev, ch)) = self.input[..word_start].char_indices().next_back() else {
                break;
            };
            if !ch.is_whitespace() {
                break;
            }
            word_start = prev;
        }

        while word_start > 0 {
            let Some((prev, ch)) = self.input[..word_start].char_indices().next_back() else {
                break;
            };
            if ch.is_whitespace() {
                break;
            }
            word_start = prev;
        }

        if word_start < cursor_byte {
            self.input.replace_range(word_start..cursor_byte, "");
            self.cursor_position = char_count(&self.input[..word_start]);
            self.slash_menu_hidden = false;
            self.mention_menu_hidden = false;
            self.mention_menu_selected = 0;
            self.needs_redraw = true;
        }
    }

    /// Delete from the cursor to the start of the line.
    pub fn delete_to_start_of_line(&mut self) {
        self.clear_input_history_navigation();
        self.selected_attachment_index = None;
        if self.cursor_position == 0 {
            return;
        }

        let cursor_byte = byte_index_at_char(&self.input, self.cursor_position);
        // Find the start of the current line (last newline or start of string)
        let line_start = self.input[..cursor_byte]
            .rfind('\n')
            .map(|idx| idx + 1)
            .unwrap_or(0);

        if line_start < cursor_byte {
            self.input.replace_range(line_start..cursor_byte, "");
            self.cursor_position = char_count(&self.input[..line_start]);
            self.slash_menu_hidden = false;
            self.mention_menu_hidden = false;
            self.mention_menu_selected = 0;
            self.needs_redraw = true;
        }
    }

    /// Delete the word after the cursor.
    pub fn delete_word_forward(&mut self) {
        self.clear_input_history_navigation();
        self.selected_attachment_index = None;
        let cursor_byte = byte_index_at_char(&self.input, self.cursor_position);
        if cursor_byte >= self.input.len() {
            return;
        }

        let mut word_end = cursor_byte;
        while word_end < self.input.len() {
            let Some(ch) = self.input[word_end..].chars().next() else {
                break;
            };
            if !ch.is_whitespace() {
                break;
            }
            word_end += ch.len_utf8();
        }

        while word_end < self.input.len() {
            let Some(ch) = self.input[word_end..].chars().next() else {
                break;
            };
            if ch.is_whitespace() {
                break;
            }
            word_end += ch.len_utf8();
        }

        if cursor_byte < word_end {
            self.input.replace_range(cursor_byte..word_end, "");
            self.slash_menu_hidden = false;
            self.mention_menu_hidden = false;
            self.mention_menu_selected = 0;
            self.needs_redraw = true;
        }
    }

    /// Cut from the cursor to the end of the current logical line into the
    /// kill buffer. If the cursor is already at end-of-line and a trailing
    /// newline exists, that newline is consumed so repeated invocations
    /// continue to make progress (matching emacs/codex semantics).
    ///
    /// Returns `true` when bytes were moved into the kill buffer.
    pub fn kill_to_end_of_line(&mut self) -> bool {
        self.clear_input_history_navigation();
        let total_chars = char_count(&self.input);
        let cursor = self.cursor_position.min(total_chars);
        let start_byte = byte_index_at_char(&self.input, cursor);

        // Find the byte offset of the next '\n' (relative to the whole string)
        // or the end of the buffer if no newline exists at/after the cursor.
        let eol_byte = self.input[start_byte..]
            .find('\n')
            .map(|rel| start_byte + rel)
            .unwrap_or_else(|| self.input.len());

        let end_byte = if start_byte == eol_byte {
            // Cursor is at EOL — consume the newline itself if one is there.
            if eol_byte < self.input.len() {
                eol_byte + 1
            } else {
                return false;
            }
        } else {
            eol_byte
        };

        let removed: String = self.input[start_byte..end_byte].to_string();
        if removed.is_empty() {
            return false;
        }

        self.kill_buffer = removed;
        self.input.replace_range(start_byte..end_byte, "");
        // Cursor stays at the same character index (start of removed range).
        self.cursor_position = cursor;
        self.slash_menu_hidden = false;
        self.mention_menu_hidden = false;
        self.mention_menu_selected = 0;
        self.needs_redraw = true;
        true
    }

    /// Insert the contents of the kill buffer at the cursor, advancing it.
    /// The kill buffer is left intact so multiple yanks duplicate the text.
    /// Returns `true` if any text was inserted.
    pub fn yank(&mut self) -> bool {
        if self.kill_buffer.is_empty() {
            return false;
        }
        self.clear_input_history_navigation();
        let text = self.kill_buffer.clone();
        let cursor = self.cursor_position.min(char_count(&self.input));
        let byte_index = byte_index_at_char(&self.input, cursor);
        self.input.insert_str(byte_index, &text);
        self.cursor_position = cursor + char_count(&text);
        self.slash_menu_hidden = false;
        self.mention_menu_hidden = false;
        self.mention_menu_selected = 0;
        self.needs_redraw = true;
        true
    }

    pub fn move_cursor_left(&mut self) {
        self.cursor_position = self.cursor_position.saturating_sub(1);
        self.needs_redraw = true;
    }

    pub fn move_cursor_right(&mut self) {
        if self.cursor_position < char_count(&self.input) {
            self.cursor_position += 1;
            self.needs_redraw = true;
        }
    }

    pub fn move_cursor_start(&mut self) {
        self.cursor_position = 0;
        self.needs_redraw = true;
    }

    pub fn move_cursor_end(&mut self) {
        self.cursor_position = char_count(&self.input);
        self.needs_redraw = true;
    }

    /// Move forward one word. Skips over the current word then any trailing
    /// whitespace to land on the first character of the next word.
    pub fn move_cursor_word_forward(&mut self) {
        let text = self.input.clone();
        let total = char_count(&text);
        let mut pos = self.cursor_position;
        if pos >= total {
            return;
        }
        // Skip non-whitespace (current word).
        while pos < total {
            let byte = byte_index_at_char(&text, pos);
            let ch = text[byte..].chars().next().unwrap_or(' ');
            if ch.is_whitespace() {
                break;
            }
            pos += 1;
        }
        // Skip whitespace.
        while pos < total {
            let byte = byte_index_at_char(&text, pos);
            let ch = text[byte..].chars().next().unwrap_or(' ');
            if !ch.is_whitespace() {
                break;
            }
            pos += 1;
        }
        self.cursor_position = pos;
        self.needs_redraw = true;
    }

    /// Move backward one word. Skips leading whitespace then the preceding
    /// word to land on its first character.
    pub fn move_cursor_word_backward(&mut self) {
        let text = self.input.clone();
        let mut pos = self.cursor_position;
        if pos == 0 {
            return;
        }
        // Step back one so we're not already at the word start.
        pos -= 1;
        // Skip whitespace.
        while pos > 0 {
            let byte = byte_index_at_char(&text, pos);
            let ch = text[byte..].chars().next().unwrap_or(' ');
            if !ch.is_whitespace() {
                break;
            }
            pos -= 1;
        }
        // Skip non-whitespace.
        while pos > 0 {
            let byte = byte_index_at_char(&text, pos - 1);
            let ch = text[byte..].chars().next().unwrap_or(' ');
            if ch.is_whitespace() {
                break;
            }
            pos -= 1;
        }
        self.cursor_position = pos;
        self.needs_redraw = true;
    }

    // === Vim composer mode helpers ===

    /// Move the cursor to the start of the current logical line (vim `0`).
    pub fn vim_move_line_start(&mut self) {
        let text = self.input.clone();
        let cursor_byte = byte_index_at_char(&text, self.cursor_position);
        // Walk backward until we find a newline or the start of the string.
        let line_start_byte = text[..cursor_byte].rfind('\n').map_or(0, |idx| idx + 1);
        self.cursor_position = char_count(&text[..line_start_byte]);
        self.needs_redraw = true;
    }

    /// Move the cursor to the end of the current logical line (vim `$`).
    pub fn vim_move_line_end(&mut self) {
        let text = self.input.clone();
        let cursor_byte = byte_index_at_char(&text, self.cursor_position);
        // Walk forward to the next newline or end-of-string.
        let line_end_char = text[cursor_byte..].find('\n').map_or_else(
            || char_count(&text),
            |rel| char_count(&text[..cursor_byte + rel]),
        );
        self.cursor_position = line_end_char;
        self.needs_redraw = true;
    }

    /// Move forward one word (vim `w`).  Skips over the current word then any
    /// trailing whitespace to land on the first character of the next word.
    pub fn vim_move_word_forward(&mut self) {
        self.move_cursor_word_forward();
    }

    /// Move backward one word (vim `b`).  Skips leading whitespace then the
    /// preceding word to land on its first character.
    pub fn vim_move_word_backward(&mut self) {
        self.move_cursor_word_backward();
    }

    /// Delete the character under the cursor (vim `x`).
    pub fn vim_delete_char_under_cursor(&mut self) {
        let total = char_count(&self.input);
        if self.cursor_position >= total {
            return;
        }
        let pos = self.cursor_position;
        remove_char_at(&mut self.input, pos);
        // Keep cursor in bounds after deletion.
        let new_total = char_count(&self.input);
        if self.cursor_position > 0 && self.cursor_position >= new_total {
            self.cursor_position = new_total.saturating_sub(1);
        }
        self.needs_redraw = true;
    }

    /// Delete the entire current logical line (vim `dd`).
    pub fn vim_delete_line(&mut self) {
        let text = self.input.clone();
        let cursor_byte = byte_index_at_char(&text, self.cursor_position);
        let line_start_byte = text[..cursor_byte].rfind('\n').map_or(0, |idx| idx + 1);
        let line_end_byte = text[cursor_byte..]
            .find('\n')
            .map_or(text.len(), |rel| cursor_byte + rel);

        // Include the trailing newline if present, or the leading newline for the
        // very last non-terminated line to avoid leaving a dangling newline.
        let (remove_start, remove_end) = if line_end_byte < text.len() {
            // There is a newline after the line — remove it too.
            (line_start_byte, line_end_byte + 1)
        } else if line_start_byte > 0 {
            // Last line without trailing newline — remove the preceding newline.
            (line_start_byte - 1, line_end_byte)
        } else {
            // Only line in the buffer.
            (line_start_byte, line_end_byte)
        };

        self.input.replace_range(remove_start..remove_end, "");
        self.cursor_position = char_count(&self.input[..remove_start]);
        self.needs_redraw = true;
    }

    /// Enter insert mode at the cursor (vim `i`).
    pub fn vim_enter_insert(&mut self) {
        self.vim_mode = VimMode::Insert;
        self.needs_redraw = true;
    }

    /// Enter insert mode after the cursor (vim `a`).
    pub fn vim_enter_append(&mut self) {
        let total = char_count(&self.input);
        if self.cursor_position < total {
            self.cursor_position += 1;
        }
        self.vim_mode = VimMode::Insert;
        self.needs_redraw = true;
    }

    /// Open a new line below and enter insert mode (vim `o`).
    pub fn vim_open_line_below(&mut self) {
        // Move to end of line, then insert a newline.
        self.vim_move_line_end();
        self.insert_char('\n');
        self.vim_mode = VimMode::Insert;
    }

    /// Return to Normal mode from Insert or Visual (vim `Esc`).
    pub fn vim_enter_normal(&mut self) {
        self.vim_mode = VimMode::Normal;
        self.vim_pending_d = false;
        // In Normal mode the cursor sits on a character, not after the last one.
        let total = char_count(&self.input);
        if self.cursor_position > 0 && self.cursor_position >= total {
            self.cursor_position = total.saturating_sub(1);
        }
        self.needs_redraw = true;
    }

    /// Returns `true` when vim mode is active and the composer is in Normal
    /// mode, which means character keys should NOT be inserted as text.
    #[must_use]
    pub fn vim_is_normal_mode(&self) -> bool {
        self.composer.vim_enabled && self.composer.vim_mode == VimMode::Normal
    }

    /// Returns `true` when vim mode is active and the composer is in Visual mode.
    #[must_use]
    pub fn vim_is_visual_mode(&self) -> bool {
        self.composer.vim_enabled && self.composer.vim_mode == VimMode::Visual
    }

    /// Move the cursor down one logical line within the buffer (vim `j`).
    /// Falls back to history-down when already on the last line.
    pub fn vim_move_down(&mut self) {
        let text = self.input.clone();
        let total = char_count(&text);
        if self.cursor_position >= total {
            self.history_down();
            return;
        }
        let cursor_byte = byte_index_at_char(&text, self.cursor_position);
        let rest = &text[cursor_byte..];
        if let Some(rel_nl) = rest.find('\n') {
            // Column offset on the current line.
            let line_start_byte = text[..cursor_byte].rfind('\n').map_or(0, |i| i + 1);
            let col = char_count(&text[line_start_byte..cursor_byte]);
            let next_line_start = cursor_byte + rel_nl + 1;
            let next_line = &text[next_line_start..];
            let next_line_len = next_line.find('\n').unwrap_or(next_line.len());
            let next_line_char_len =
                char_count(&text[next_line_start..next_line_start + next_line_len]);
            let target_col = col.min(next_line_char_len);
            self.cursor_position = char_count(&text[..next_line_start]) + target_col;
            self.needs_redraw = true;
        } else {
            self.history_down();
        }
    }

    /// Move the cursor up one logical line within the buffer (vim `k`).
    /// Falls back to history-up when already on the first line.
    pub fn vim_move_up(&mut self) {
        let text = self.input.clone();
        let cursor_byte = byte_index_at_char(&text, self.cursor_position);
        if let Some(prev_nl) = text[..cursor_byte].rfind('\n') {
            // Column on the current line.
            let line_start_byte = prev_nl + 1;
            let col = char_count(&text[line_start_byte..cursor_byte]);
            // Find start of the previous line.
            let prev_line_end = prev_nl; // byte of the newline itself
            let prev_start = text[..prev_line_end].rfind('\n').map_or(0, |i| i + 1);
            let prev_line_len = char_count(&text[prev_start..prev_line_end]);
            let target_col = col.min(prev_line_len);
            self.cursor_position = char_count(&text[..prev_start]) + target_col;
            self.needs_redraw = true;
        } else {
            self.history_up();
        }
    }

    pub fn clear_input(&mut self) {
        self.clear_input_history_navigation();
        self.input.clear();
        self.cursor_position = 0;
        self.selected_attachment_index = None;
        self.slash_menu_selected = 0;
        self.slash_menu_hidden = false;
        self.paste_burst.clear_after_explicit_paste();
        self.needs_redraw = true;
    }

    pub fn clear_input_recoverable(&mut self) {
        self.stash_current_input_for_recovery();
        self.clear_input();
    }

    pub fn stash_current_input_for_recovery(&mut self) {
        let draft = self.input.clone();
        self.remember_draft_for_recovery(draft);
    }

    fn remember_draft_for_recovery(&mut self, draft: String) {
        if draft.trim().is_empty() {
            return;
        }
        self.draft_history.retain(|existing| existing != &draft);
        self.draft_history.push_back(draft);
        while self.draft_history.len() > MAX_DRAFT_HISTORY {
            let _ = self.draft_history.pop_front();
        }
    }

    pub fn start_history_search(&mut self) {
        if self.composer_history_search.is_some() {
            return;
        }
        self.composer_history_search = Some(ComposerHistorySearch::new(
            self.input.clone(),
            self.cursor_position,
        ));
        self.slash_menu_hidden = true;
        self.mention_menu_hidden = true;
        self.paste_burst.clear_after_explicit_paste();
        self.status_message = Some("History search: type to filter, Enter accepts".to_string());
        self.needs_redraw = true;
    }

    pub fn is_history_search_active(&self) -> bool {
        self.composer_history_search.is_some()
    }

    pub fn history_search_query(&self) -> Option<&str> {
        self.composer_history_search
            .as_ref()
            .map(|search| search.query.as_str())
    }

    pub fn history_search_selected_index(&self) -> usize {
        self.composer_history_search
            .as_ref()
            .map_or(0, |search| search.selected)
    }

    pub fn composer_display_input(&self) -> &str {
        self.history_search_query().unwrap_or(&self.input)
    }

    pub fn composer_display_cursor(&self) -> usize {
        self.composer_history_search
            .as_ref()
            .map_or(self.cursor_position, |search| char_count(&search.query))
    }

    pub fn history_search_matches(&self) -> Vec<String> {
        let Some(query) = self.history_search_query() else {
            return Vec::new();
        };
        self.history_search_matches_for_query(query)
    }

    fn history_search_matches_for_query(&self, query: &str) -> Vec<String> {
        let normalized_query = query.trim().to_lowercase();
        let mut seen: HashSet<&str> = HashSet::new();
        let mut matches = Vec::new();

        for candidate in self
            .draft_history
            .iter()
            .rev()
            .chain(self.input_history.iter().rev())
        {
            if candidate.trim().is_empty() || !seen.insert(candidate.as_str()) {
                continue;
            }
            if normalized_query.is_empty() || candidate.to_lowercase().contains(&normalized_query) {
                matches.push(candidate.clone());
            }
        }

        matches
    }

    fn clamp_history_search_selection(&mut self) {
        let Some(search) = self.composer_history_search.as_ref() else {
            return;
        };
        let selected = search.selected;
        let query = search.query.clone();
        let match_count = self.history_search_matches_for_query(&query).len();
        if let Some(search) = self.composer_history_search.as_mut() {
            search.selected = if match_count == 0 {
                0
            } else {
                selected.min(match_count.saturating_sub(1))
            };
        }
    }

    pub fn history_search_insert_char(&mut self, ch: char) {
        if let Some(search) = self.composer_history_search.as_mut() {
            search.query.push(ch);
            search.selected = 0;
            self.status_message = Some("History search: Enter accepts, Esc restores".to_string());
            self.needs_redraw = true;
        }
    }

    pub fn history_search_insert_str(&mut self, text: &str) {
        if text.is_empty() {
            return;
        }
        if let Some(search) = self.composer_history_search.as_mut() {
            search.query.push_str(&normalize_paste_text(text));
            search.selected = 0;
            self.status_message = Some("History search: Enter accepts, Esc restores".to_string());
            self.needs_redraw = true;
        }
    }

    pub fn history_search_backspace(&mut self) {
        if let Some(search) = self.composer_history_search.as_mut() {
            search.query.pop();
            search.selected = 0;
            self.needs_redraw = true;
        }
        self.clamp_history_search_selection();
    }

    pub fn history_search_select_previous(&mut self) {
        if let Some(search) = self.composer_history_search.as_mut() {
            search.selected = search.selected.saturating_sub(1);
            self.needs_redraw = true;
        }
    }

    pub fn history_search_select_next(&mut self) {
        let Some(search) = self.composer_history_search.as_ref() else {
            return;
        };
        let query = search.query.clone();
        let selected = search.selected;
        let match_count = self.history_search_matches_for_query(&query).len();
        if let Some(search) = self.composer_history_search.as_mut()
            && match_count > 0
        {
            search.selected = (selected + 1).min(match_count.saturating_sub(1));
            self.needs_redraw = true;
        }
    }

    pub fn accept_history_search(&mut self) -> bool {
        let Some(search) = self.composer_history_search.take() else {
            return false;
        };
        let matches = self.history_search_matches_for_query(&search.query);
        if let Some(selected) = matches
            .get(search.selected.min(matches.len().saturating_sub(1)))
            .cloned()
        {
            self.input = selected;
            self.cursor_position = char_count(&self.input);
            self.history_index = None;
            self.status_message = Some("History match inserted into composer".to_string());
            self.needs_redraw = true;
            true
        } else {
            self.composer_history_search = Some(search);
            self.status_message = Some("No history matches".to_string());
            self.needs_redraw = true;
            false
        }
    }

    pub fn cancel_history_search(&mut self) {
        let Some(search) = self.composer_history_search.take() else {
            return;
        };
        self.input = search.pre_search_input;
        self.cursor_position = search.pre_search_cursor.min(char_count(&self.input));
        self.status_message = Some("History search canceled".to_string());
        self.needs_redraw = true;
    }

    pub fn submit_input(&mut self) -> Option<String> {
        if self.input.trim().is_empty() {
            self.paste_burst.clear_after_explicit_paste();
            return None;
        }
        // Safety net: if any earlier path filled the buffer above the
        // safety cap without going through `insert_paste_text`, fold it
        // into a workspace paste file now (#553). Bracketed pastes hit
        // the consolidation in `insert_paste_text` first, so the user
        // sees the @mention in the composer before submission.
        self.consolidate_large_input_if_oversized();
        let input = self.input.clone();
        if !input.starts_with('/') {
            self.input_history.push(input.clone());
            if self.max_input_history == 0 {
                self.input_history.clear();
            } else if self.input_history.len() > self.max_input_history {
                let excess = self.input_history.len() - self.max_input_history;
                self.input_history.drain(0..excess);
            }
            // Mirror to the persisted cross-session history (#366) so
            // arrow-up recall works across restarts. Best-effort write —
            // see `composer_history::append_history` for failure modes.
            crate::composer_history::append_history(&input);
        }
        self.history_index = None;
        self.history_navigation_draft = None;
        self.clear_input();
        Some(input)
    }

    /// Composer-Enter dispatch. Returns `Some(input)` when the press should
    /// fire a submit; `None` when Enter was absorbed (paste-burst Enter
    /// suppression — see #1073).
    ///
    /// Two suppression cases are handled here. Both are silent: nothing
    /// visible happens beyond the text gaining a newline.
    ///
    /// 1. **Burst active.** A paste burst is currently being assembled in
    ///    `paste_burst.buffer`. The Enter is part of the paste content;
    ///    append `\n` to the buffer so the next flush includes it, do not
    ///    submit, and extend the suppression window so a follow-on Enter
    ///    (i.e. the *next* line of a multi-line paste) is also absorbed.
    /// 2. **Window open after flush.** A burst just flushed into
    ///    `self.input`, but the suppression window is still alive. The
    ///    Enter is the trailing newline of that paste, not a submit gesture
    ///    by the user. Insert `\n` directly into the composer text and
    ///    re-arm the window.
    ///
    /// Outside both cases the call falls through to [`Self::submit_input`]
    /// unchanged so normal Enter-to-send behaviour is preserved.
    pub fn handle_composer_enter(&mut self) -> Option<String> {
        if self.use_paste_burst_detection {
            let now = Instant::now();
            if self
                .paste_burst
                .newline_should_insert_instead_of_submit(now)
            {
                if !self.paste_burst.append_newline_if_active(now) {
                    self.insert_char('\n');
                    self.paste_burst.extend_window(now);
                }
                self.needs_redraw = true;
                return None;
            }
        }
        self.submit_input()
    }

    /// Public wrapper around [`Self::consolidate_large_input`] that no-ops
    /// when the current input fits inside the safety cap. Both the paste-
    /// insert path (visible-before-submit) and the submit-time safety net
    /// route through here, so the cap is enforced exactly once even when
    /// both paths fire on the same buffer.
    fn consolidate_large_input_if_oversized(&mut self) {
        if char_count(&self.input) > MAX_SUBMITTED_INPUT_CHARS {
            self.consolidate_large_input();
        }
    }

    /// When the composer input exceeds [`MAX_SUBMITTED_INPUT_CHARS`], write
    /// the full content to a timestamped paste file under
    /// `.deepseek/pastes/` and replace `self.input` with an `@`-mention
    /// pointing at it so the model can read the full content via the
    /// normal file-mention resolution path (#553).
    fn consolidate_large_input(&mut self) {
        let full_input = std::mem::take(&mut self.input);
        self.cursor_position = 0;

        let now = chrono::Local::now();
        let suffix = uuid::Uuid::new_v4().to_string()[..8].to_string();
        let filename = format!("paste-{}-{}.md", now.format("%Y-%m-%d-%H%M%S"), suffix);
        let rel_path = format!(".deepseek/pastes/{filename}");

        let pastes_dir = self.workspace.join(".deepseek/pastes");
        if let Err(e) = std::fs::create_dir_all(&pastes_dir) {
            // Fallback: keep a truncated version so we don't lose the
            // user's input entirely when the filesystem is unhappy.
            self.input = full_input.chars().take(MAX_SUBMITTED_INPUT_CHARS).collect();
            self.cursor_position = char_count(&self.input);
            self.push_status_toast(
                format!("Failed to create paste directory: {e}"),
                StatusToastLevel::Error,
                Some(8_000),
            );
            return;
        }

        let file_path = self.workspace.join(&rel_path);
        if let Err(e) = std::fs::write(&file_path, &full_input) {
            self.input = full_input.chars().take(MAX_SUBMITTED_INPUT_CHARS).collect();
            self.cursor_position = char_count(&self.input);
            self.push_status_toast(
                format!("Failed to write paste file: {e}"),
                StatusToastLevel::Error,
                Some(8_000),
            );
            return;
        }

        self.input = format!("@{rel_path}");
        self.cursor_position = char_count(&self.input);
        self.push_status_toast(
            "Large paste consolidated — sent as @mention",
            StatusToastLevel::Info,
            Some(5_000),
        );
    }

    pub fn queue_message(&mut self, message: QueuedMessage) {
        self.queued_messages.push_back(message);
    }

    pub fn pop_queued_message(&mut self) -> Option<QueuedMessage> {
        self.queued_messages.pop_front()
    }

    pub fn remove_queued_message(&mut self, index: usize) -> Option<QueuedMessage> {
        self.queued_messages.remove(index)
    }

    pub fn queued_message_count(&self) -> usize {
        self.queued_messages.len()
    }

    /// Pop the most-recently queued message back into the composer for editing
    /// (issue #85 — ↑ affordance). The popped message is parked in
    /// [`Self::queued_draft`] so the next Enter re-queues it carrying its
    /// original skill instruction. No-op if the composer already has typed
    /// content or a draft is already being edited — surfacing the affordance
    /// would be ambiguous in either case.
    ///
    /// Returns `true` when the composer state was mutated.
    pub fn pop_last_queued_into_draft(&mut self) -> bool {
        if !self.input.is_empty() || self.queued_draft.is_some() {
            return false;
        }
        let Some(msg) = self.queued_messages.pop_back() else {
            return false;
        };
        self.input = msg.display.clone();
        self.cursor_position = char_count(&self.input);
        self.selected_attachment_index = None;
        self.queued_draft = Some(msg);
        self.needs_redraw = true;
        true
    }

    /// Park a legacy pending steer. New keyboard handling routes running-turn
    /// drafts through Enter (same-turn steer) or Tab (next-turn follow-up).
    #[allow(dead_code)]
    pub fn push_pending_steer(&mut self, message: QueuedMessage) {
        self.pending_steers.push_back(message);
        self.submit_pending_steers_after_interrupt = true;
        self.needs_redraw = true;
    }

    /// Drain the pending-steer queue and clear the resend flag. Returns the
    /// messages in submit order (oldest first).
    pub fn drain_pending_steers(&mut self) -> Vec<QueuedMessage> {
        self.submit_pending_steers_after_interrupt = false;
        if self.pending_steers.is_empty() {
            return Vec::new();
        }
        self.needs_redraw = true;
        self.pending_steers.drain(..).collect()
    }

    /// Decide how to route a fresh composer submit.
    ///
    /// #382: default to Queue when busy — the user shouldn't have to distinguish
    /// "streaming" from "tool execution". Ctrl+Enter overrides to Steer.
    ///
    /// Truth table:
    ///   offline=F, busy=F → Immediate
    ///   offline=F, busy=T → Queue  (was Steer for non-streaming; now unified)
    ///   offline=T, busy=* → Queue
    #[must_use]
    pub fn decide_submit_disposition(&self) -> SubmitDisposition {
        if self.offline_mode {
            return SubmitDisposition::Queue;
        }
        if !self.is_loading {
            return SubmitDisposition::Immediate;
        }
        // Busy: always queue. Ctrl+Enter routes through steer_user_message directly.
        SubmitDisposition::Queue
    }

    /// Mark the in-flight streaming Assistant cell as interrupted: prepend
    /// `[interrupted]` to whatever streamed so far (so the user can see what
    /// was salvaged) and flip `streaming` off so the spinner halts. No-op if
    /// no Assistant cell is currently streaming.
    ///
    /// Deliberate divergence from openai/codex which discards partial output
    /// on abort — V4 thinking is expensive and the user usually wants to see
    /// what the model produced before steering.
    pub fn finalize_streaming_assistant_as_interrupted(&mut self) {
        let Some(index) = self.streaming_message_index.take() else {
            return;
        };
        if let Some(HistoryCell::Assistant { content, streaming }) = self.history.get_mut(index) {
            *streaming = false;
            if content.is_empty() {
                *content = "[interrupted]".to_string();
            } else if !content.starts_with("[interrupted]") {
                content.insert_str(0, "[interrupted] ");
            }
        }
        self.bump_history_cell(index);
    }

    pub fn history_up(&mut self) {
        if self.input_history.is_empty() {
            return;
        }
        if self.history_index.is_none() {
            self.history_navigation_draft = Some(InputHistoryDraft {
                input: self.input.clone(),
                cursor: self.cursor_position,
            });
        }
        let new_index = match self.history_index {
            None => self.input_history.len().saturating_sub(1),
            Some(i) => i.saturating_sub(1),
        };
        self.history_index = Some(new_index);
        self.input = self.input_history[new_index].clone();
        self.cursor_position = char_count(&self.input);
        self.selected_attachment_index = None;
        self.slash_menu_hidden = false;
        self.paste_burst.clear_after_explicit_paste();
    }

    pub fn history_down(&mut self) {
        if self.input_history.is_empty() {
            return;
        }
        match self.history_index {
            None => {}
            Some(i) => {
                if i + 1 < self.input_history.len() {
                    self.history_index = Some(i + 1);
                    self.input = self.input_history[i + 1].clone();
                    self.cursor_position = char_count(&self.input);
                    self.selected_attachment_index = None;
                    self.slash_menu_hidden = false;
                    self.paste_burst.clear_after_explicit_paste();
                } else {
                    self.history_index = None;
                    if let Some(draft) = self.history_navigation_draft.take() {
                        self.input = draft.input;
                        self.cursor_position = draft.cursor.min(char_count(&self.input));
                        self.selected_attachment_index = None;
                        self.slash_menu_hidden = false;
                        self.paste_burst.clear_after_explicit_paste();
                        self.needs_redraw = true;
                    } else {
                        self.clear_input();
                    }
                }
            }
        }
    }

    fn clear_input_history_navigation(&mut self) {
        self.history_index = None;
        self.history_navigation_draft = None;
    }

    /// Retry a `try_lock` up to `retries` times with a 1ms pause between
    /// attempts. Returns `Some(guard)` on success, `None` if the lock
    /// remains contended after all retries.
    fn retry_lock<T>(
        mutex: &tokio::sync::Mutex<T>,
        retries: u32,
    ) -> Option<tokio::sync::MutexGuard<'_, T>> {
        for _ in 0..retries {
            if let Ok(guard) = mutex.try_lock() {
                return Some(guard);
            }
            std::thread::sleep(std::time::Duration::from_millis(1));
        }
        None
    }

    pub fn clear_todos(&mut self) -> bool {
        // Clear the todo list (the sidebar checklist). Retry with try_lock
        // so /clear always resets todos even when the engine briefly holds
        // the mutex during tool execution.
        let todos_cleared = if let Some(mut todos) = Self::retry_lock(&self.todos, 100) {
            todos.clear();
            true
        } else {
            false
        };
        // Also clear the plan state — /clear means a full reset.
        if let Some(mut plan) = Self::retry_lock(&self.plan_state, 100) {
            *plan = crate::tools::plan::PlanState::default();
        }
        todos_cleared
    }

    pub fn update_model_compaction_budget(&mut self) {
        let model = self.effective_model_for_budget().to_string();
        self.compact_threshold =
            compaction_threshold_for_model_and_effort(&model, self.reasoning_effort.api_value());
    }

    pub fn effective_model_for_budget(&self) -> &str {
        if self.auto_model {
            return self
                .last_effective_model
                .as_deref()
                .filter(|model| *model != "auto")
                .unwrap_or(DEFAULT_TEXT_MODEL);
        }
        &self.model
    }

    pub fn model_display_label(&self) -> String {
        if self.auto_model {
            if let Some(effective) = self.last_effective_model.as_deref()
                && effective != "auto"
            {
                return format!("auto: {effective}");
            }
            return "auto".to_string();
        }
        self.model.clone()
    }

    pub fn reasoning_effort_display_label(&self) -> String {
        if self.auto_model || self.reasoning_effort == ReasoningEffort::Auto {
            if let Some(effective) = self.last_effective_reasoning_effort {
                return format!("auto: {}", effective.short_label());
            }
            return "auto".to_string();
        }
        self.reasoning_effort.short_label().to_string()
    }

    pub fn compaction_config(&self) -> CompactionConfig {
        CompactionConfig {
            enabled: self.auto_compact,
            token_threshold: self.compact_threshold,
            model: self.model.clone(),
            ..Default::default()
        }
    }

    /// Forward the active cycle configuration to the engine. Cloned so the
    /// engine has its own copy to mutate per-session.
    pub fn cycle_config(&self) -> CycleConfig {
        self.cycle.clone()
    }
}

