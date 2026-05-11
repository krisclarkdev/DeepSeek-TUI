use super::state::*;
use super::mutators::*;
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

pub fn media_attachment_reference(kind: &str, path: &Path, description: Option<&str>) -> String {
    match description {
        Some(description) if !description.trim().is_empty() => {
            format!(
                "[Attached {kind}: {} at {}]",
                description.trim(),
                path.display()
            )
        }
        _ => format!("[Attached {kind}: {}]", path.display()),
    }
}

// === Actions ===

/// Actions emitted by the UI event loop.
#[derive(Debug, Clone, PartialEq)]
pub enum AppAction {
    Quit,
    #[allow(dead_code)] // For explicit /save command
    SaveSession(PathBuf),
    #[allow(dead_code)] // For explicit /load command
    LoadSession(PathBuf),
    SyncSession {
        session_id: Option<String>,
        messages: Vec<Message>,
        system_prompt: Option<SystemPrompt>,
        model: String,
        workspace: PathBuf,
    },
    OpenConfigEditor(ConfigUiMode),
    OpenConfigView,
    /// Open the `/model` two-pane picker (Pro/Flash + Off/High/Max).
    OpenModelPicker,
    /// Open the `/provider` picker modal — DeepSeek / NVIDIA NIM / OpenRouter
    /// / Novita with inline API-key prompt for un-configured providers (#52).
    OpenProviderPicker,
    /// Open the `/mode` picker modal for Agent / Plan / YOLO.
    OpenModePicker,
    /// Open the `/statusline` multi-select picker for footer items.
    OpenStatusPicker,
    /// Open the `/feedback` picker for GitHub issue/security destinations.
    OpenFeedbackPicker,
    /// Open an external URL in the system browser.
    OpenExternalUrl {
        url: String,
        label: String,
    },
    /// Send a message to the AI (normal chat mode).
    SendMessage(String),
    /// Run a Recursive Language Model (RLM) turn — Algorithm 1 from
    /// Zhang et al. (arXiv:2512.24601). The prompt is stored in the REPL;
    /// the root LLM only sees metadata.
    Rlm {
        /// The user's prompt — stored in REPL, NOT in LLM context.
        prompt: String,
        /// Model for the root LLM.
        model: String,
        /// Model for sub-LLM (llm_query) calls.
        child_model: String,
        /// Recursion budget for `sub_rlm()` calls.
        max_depth: u32,
    },
    ListSubAgents,
    FetchModels,
    CacheWarmup,
    /// Switch the active LLM backend (DeepSeek vs NVIDIA NIM) without
    /// restarting the process. The runtime rebuilds its API client from
    /// the updated config. `model` overrides the post-switch model
    /// (already normalized but not yet provider-prefixed).
    SwitchProvider {
        provider: ApiProvider,
        model: Option<String>,
    },
    UpdateCompaction(CompactionConfig),
    OpenContextInspector,
    CompactContext,
    TaskAdd {
        prompt: String,
    },
    TaskList,
    TaskShow {
        id: String,
    },
    TaskCancel {
        id: String,
    },
    ShellJob(ShellJobAction),
    Mcp(McpUiAction),
    /// Switch to a different config profile without restarting.
    SwitchProfile {
        /// Profile name to load.
        profile: String,
    },
    /// Export and share the current session as a web URL.
    ShareSession {
        history_len: usize,
        model: String,
        mode: String,
    },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ShellJobAction {
    List,
    Show {
        id: String,
    },
    Poll {
        id: String,
        wait: bool,
    },
    SendStdin {
        id: String,
        input: String,
        close: bool,
    },
    Cancel {
        id: String,
    },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum McpUiAction {
    Show,
    Init {
        force: bool,
    },
    AddStdio {
        name: String,
        command: String,
        args: Vec<String>,
    },
    AddHttp {
        name: String,
        url: String,
    },
    Enable {
        name: String,
    },
    Disable {
        name: String,
    },
    Remove {
        name: String,
    },
    Validate,
    Reload,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::Config;
    use crate::tools::plan::{PlanItemArg, StepStatus, UpdatePlanArgs};
    use crate::tools::todo::TodoStatus;
    use crate::tui::clipboard::PastedImage;

    fn test_options(yolo: bool) -> TuiOptions {
        TuiOptions {
            model: "test-model".to_string(),
            workspace: PathBuf::from("."),
            config_path: None,
            config_profile: None,
            allow_shell: yolo,
            use_alt_screen: true,
            use_mouse_capture: false,
            use_bracketed_paste: true,
            max_subagents: 1,
            skills_dir: PathBuf::from("."),
            memory_path: PathBuf::from("memory.md"),
            notes_path: PathBuf::from("notes.txt"),
            mcp_config_path: PathBuf::from("mcp.json"),
            use_memory: false,
            start_in_agent_mode: yolo,
            skip_onboarding: false,
            yolo,
            resume_session_id: None,
            initial_input: None,
        }
    }

    #[test]
    fn test_trust_mode_follows_yolo_on_startup() {
        let app = App::new(test_options(true), &Config::default());
        assert!(app.trust_mode);
    }

    #[test]
    fn onboarded_user_still_gets_workspace_trust_prompt_when_needed() {
        assert_eq!(
            initial_onboarding_state(false, true, false, true),
            OnboardingState::TrustDirectory
        );
    }

    #[test]
    fn new_caches_workspace_skills_for_slash_menu() {
        let tmp = tempfile::TempDir::new().expect("tempdir");
        let workspace = tmp.path().join("workspace");
        let skill_dir = workspace.join(".agents").join("skills").join("local-skill");
        std::fs::create_dir_all(&skill_dir).expect("skill dir");
        std::fs::write(
            skill_dir.join("SKILL.md"),
            "---\nname: local-skill\ndescription: Local workspace skill\n---\nUse the local skill.\n",
        )
        .expect("skill file");

        let mut options = test_options(false);
        options.workspace = workspace.clone();
        options.skills_dir = tmp.path().join("global-skills");
        let app = App::new(options, &Config::default());

        assert_eq!(app.skills_dir, workspace.join(".agents").join("skills"));
        assert!(app.cached_skills.iter().any(|(name, description)| {
            name == "local-skill" && description == "Local workspace skill"
        }));
    }

    #[test]
    fn cached_skills_merges_across_candidate_directories() {
        let tmp = tempfile::TempDir::new().expect("tempdir");
        let workspace = tmp.path().join("workspace");

        // Higher-precedence directory contains a stale empty dir for `foo`
        // (no SKILL.md). This used to shadow the real definition further
        // down the candidate list when the cache only scanned a single dir.
        std::fs::create_dir_all(workspace.join(".agents").join("skills").join("foo"))
            .expect("stale empty dir");

        // Lower-precedence directory has the real skill.
        let real_dir = workspace.join(".claude").join("skills").join("foo");
        std::fs::create_dir_all(&real_dir).expect("real skill dir");
        std::fs::write(
            real_dir.join("SKILL.md"),
            "---\nname: foo\ndescription: Real foo skill\n---\nbody\n",
        )
        .expect("skill file");

        let mut options = test_options(false);
        options.workspace = workspace.clone();
        options.skills_dir = tmp.path().join("global-skills");
        let app = App::new(options, &Config::default());

        assert!(
            app.cached_skills
                .iter()
                .any(|(name, description)| name == "foo" && description == "Real foo skill"),
            "cached_skills should fall through to lower-precedence dir when higher-precedence one has an empty stub: {:?}",
            app.cached_skills,
        );
    }

    #[test]
    fn paste_consolidates_oversized_text_into_paste_file_visibly() {
        // Visible-before-submit consolidation (paste UX): when a single
        // bracketed paste exceeds the safety cap, the @mention must
        // replace the input *immediately*, so the user sees what's
        // about to be sent before pressing Enter — not as a side effect
        // of submit.
        let tmp = tempfile::TempDir::new().expect("tempdir");
        let mut opts = test_options(false);
        opts.workspace = tmp.path().to_path_buf();
        let mut app = App::new(opts, &Config::default());
        let full_content = "y".repeat(MAX_SUBMITTED_INPUT_CHARS + 256);

        app.insert_paste_text(&full_content);

        // Composer should now contain the @mention, not the full text.
        assert!(
            app.input.starts_with("@.deepseek/pastes/paste-") && app.input.ends_with(".md"),
            "expected @mention in composer after large paste, got: {}",
            app.input
        );
        // The cursor moves to the end of the @mention.
        assert_eq!(app.cursor_position, app.input.chars().count());
        // The paste file must exist with the full content.
        let rel_path = &app.input[1..];
        let abs = tmp.path().join(rel_path);
        assert!(abs.is_file(), "paste file must exist at {abs:?}");
        let written = std::fs::read_to_string(&abs).expect("read");
        assert_eq!(written, full_content);
        // A toast confirms what happened so the user isn't surprised.
        assert!(
            app.status_toasts
                .iter()
                .any(|t| t.text.contains("consolidated")),
            "expected consolidation toast"
        );
    }

    #[test]
    fn paste_under_threshold_does_not_consolidate() {
        // Negative path: a small paste must NOT spawn a paste file. The
        // input stays inline so the user can edit it freely.
        let tmp = tempfile::TempDir::new().expect("tempdir");
        let mut opts = test_options(false);
        opts.workspace = tmp.path().to_path_buf();
        let mut app = App::new(opts, &Config::default());
        let small = "hello world\nthis is fine".to_string();

        app.insert_paste_text(&small);

        assert_eq!(app.input, small);
        assert!(!app.input.starts_with("@.deepseek/pastes/"));
        // No paste file gets written for under-cap pastes.
        let pastes_dir = tmp.path().join(".deepseek/pastes");
        assert!(
            !pastes_dir.exists() || std::fs::read_dir(&pastes_dir).unwrap().next().is_none(),
            "no paste file should be written for under-cap content"
        );
    }

    #[test]
    fn submit_input_consolidates_oversized_input_into_paste_file() {
        let tmp = tempfile::TempDir::new().expect("tempdir");
        let mut opts = test_options(false);
        opts.workspace = tmp.path().to_path_buf();
        let mut app = App::new(opts, &Config::default());
        let full_content = "x".repeat(MAX_SUBMITTED_INPUT_CHARS + 128);
        app.input = full_content.clone();
        app.cursor_position = app.input.chars().count();

        let submitted = app.submit_input().expect("expected submitted input");

        // The submitted text should be the @mention, not the truncated
        // original (#553).
        assert!(
            submitted.starts_with("@.deepseek/pastes/paste-"),
            "expected @mention, got: {submitted}"
        );
        assert!(
            submitted.ends_with(".md"),
            "expected .md extension, got: {submitted}"
        );

        // The paste file must exist on disk with the full original content.
        let rel_path = &submitted[1..]; // strip leading '@'
        let abs_path = tmp.path().join(rel_path);
        assert!(abs_path.is_file(), "paste file must exist at {abs_path:?}");
        let written = std::fs::read_to_string(&abs_path).expect("read paste file");
        assert_eq!(written, full_content);

        // A status toast should have been pushed.
        assert!(
            app.status_toasts
                .iter()
                .any(|toast| toast.text.contains("consolidated")),
            "expected consolidation toast, got: {:?}",
            app.status_toasts
                .iter()
                .map(|t| &t.text)
                .collect::<Vec<_>>()
        );

        // The composer must be clear after submit.
        assert!(app.input.is_empty());
    }

    #[test]
    fn app_starts_without_seeded_transcript_messages() {
        let app = App::new(test_options(false), &Config::default());
        assert!(app.history.is_empty());
        assert_eq!(app.history_version, 0);
    }

    #[test]
    fn clear_todos_resets_todos_list() {
        let mut app = App::new(test_options(false), &Config::default());

        // Seed some todos.
        {
            let mut todos = app.todos.try_lock().expect("todos lock");
            todos.add("buy milk".to_string(), TodoStatus::Pending);
            todos.add("write code".to_string(), TodoStatus::InProgress);
            assert_eq!(todos.snapshot().items.len(), 2);
        }

        assert!(app.clear_todos());

        let todos = app.todos.try_lock().expect("todos lock");
        assert!(todos.snapshot().items.is_empty());
    }

    #[test]
    fn clear_todos_resets_plan_state() {
        let mut app = App::new(test_options(false), &Config::default());

        {
            let mut plan = app
                .plan_state
                .try_lock()
                .expect("plan lock should be available");
            plan.update(UpdatePlanArgs {
                explanation: Some("test plan".to_string()),
                plan: vec![PlanItemArg {
                    step: "step 1".to_string(),
                    status: StepStatus::InProgress,
                }],
            });
            assert!(!plan.is_empty());
        }

        assert!(app.clear_todos());

        let plan = app
            .plan_state
            .try_lock()
            .expect("plan lock should be available");
        assert!(plan.is_empty());
    }

    #[test]
    fn test_cycle_mode_transitions() {
        let mut app = App::new(test_options(false), &Config::default());
        // Default mode should be Agent based on settings
        let initial_mode = app.mode;
        app.cycle_mode();
        // Mode should have changed
        assert_ne!(app.mode, initial_mode);
    }

    #[test]
    fn test_cycle_mode_reverse_transitions() {
        let mut app = App::new(test_options(false), &Config::default());

        app.mode = AppMode::Plan;
        app.cycle_mode_reverse();
        assert_eq!(app.mode, AppMode::Yolo);

        app.mode = AppMode::Agent;
        app.cycle_mode_reverse();
        assert_eq!(app.mode, AppMode::Plan);
    }

    #[test]
    fn test_clear_input() {
        let mut app = App::new(test_options(false), &Config::default());
        app.input = "test input".to_string();
        app.cursor_position = app.input.len();
        app.clear_input();
        assert!(app.input.is_empty());
        assert_eq!(app.cursor_position, 0);
    }

    #[test]
    fn test_queue_message() {
        let mut app = App::new(test_options(false), &Config::default());
        app.queue_message(QueuedMessage::new("test message".to_string(), None));
        assert_eq!(app.queued_message_count(), 1);
        assert!(app.queued_messages.front().is_some());
    }

    #[test]
    fn test_remove_queued_message() {
        let mut app = App::new(test_options(false), &Config::default());
        app.queue_message(QueuedMessage::new("first".to_string(), None));
        app.queue_message(QueuedMessage::new("second".to_string(), None));

        // Remove first (index 0)
        let removed = app.remove_queued_message(0);
        assert!(removed.is_some());
        assert_eq!(app.queued_message_count(), 1);

        // Remove second (now at index 0)
        let removed = app.remove_queued_message(0);
        assert!(removed.is_some());
        assert_eq!(app.queued_message_count(), 0);
    }

    #[test]
    fn test_remove_queued_message_invalid_index() {
        let mut app = App::new(test_options(false), &Config::default());
        app.queue_message(QueuedMessage::new("test".to_string(), None));

        // Try to remove non-existent index
        let removed = app.remove_queued_message(100);
        assert!(removed.is_none());
    }

    #[test]
    fn test_set_mode_updates_state() {
        let mut app = App::new(test_options(false), &Config::default());
        let initial_mode = app.mode;
        app.set_mode(AppMode::Yolo);
        assert_eq!(app.mode, AppMode::Yolo);
        assert_ne!(app.mode, initial_mode);
        // Yolo mode should enable trust and shell
        assert!(app.trust_mode);
        assert!(app.allow_shell);
    }

    #[test]
    fn app_new_respects_allow_shell_option_when_not_yolo() {
        let mut options = test_options(false);
        options.allow_shell = false;
        options.start_in_agent_mode = true; // avoid coupling to settings.default_mode
        let app = App::new(options, &Config::default());
        assert!(!app.allow_shell);
    }

    #[test]
    fn set_mode_yolo_restores_previous_policies_on_exit() {
        let mut options = test_options(false);
        options.allow_shell = false;
        options.start_in_agent_mode = true; // avoid coupling to settings.default_mode
        let mut app = App::new(options, &Config::default());
        app.allow_shell = false;
        app.trust_mode = false;
        app.approval_mode = ApprovalMode::Never;

        app.set_mode(AppMode::Yolo);
        assert!(app.allow_shell);
        assert!(app.trust_mode);
        assert_eq!(app.approval_mode, ApprovalMode::Auto);

        app.set_mode(AppMode::Agent);
        assert!(!app.allow_shell);
        assert!(!app.trust_mode);
        assert_eq!(app.approval_mode, ApprovalMode::Never);
    }

    #[test]
    fn leaving_yolo_after_startup_restores_baseline_policies() {
        let config = Config {
            allow_shell: Some(false),
            ..Default::default()
        };

        let mut app = App::new(test_options(true), &config);
        assert_eq!(app.mode, AppMode::Yolo);
        assert!(app.allow_shell);
        assert!(app.trust_mode);
        assert_eq!(app.approval_mode, ApprovalMode::Auto);

        app.set_mode(AppMode::Agent);
        assert!(!app.allow_shell);
        assert!(!app.trust_mode);
        assert_eq!(app.approval_mode, ApprovalMode::Suggest);
    }

    #[test]
    fn configured_approval_policy_initializes_live_approval_mode() {
        let config = Config {
            approval_policy: Some("never".to_string()),
            ..Default::default()
        };
        let mut options = test_options(false);
        options.start_in_agent_mode = true;

        let app = App::new(options, &config);

        assert_eq!(app.mode, AppMode::Agent);
        assert_eq!(app.approval_mode, ApprovalMode::Never);
    }

    #[test]
    fn test_mark_history_updated() {
        let mut app = App::new(test_options(false), &Config::default());
        let initial_version = app.history_version;
        app.mark_history_updated();
        assert!(app.history_version > initial_version);
    }

    #[test]
    fn test_scroll_operations() {
        let mut app = App::new(test_options(false), &Config::default());
        // Just verify scroll methods can be called without panic
        app.scroll_up(5);
        app.scroll_down(3);
    }

    #[test]
    fn test_add_message() {
        let mut app = App::new(test_options(false), &Config::default());
        let initial_len = app.history.len();
        app.add_message(HistoryCell::User {
            content: "test".to_string(),
        });
        assert_eq!(app.history.len(), initial_len + 1);
    }

    #[test]
    fn test_compaction_config() {
        let app = App::new(test_options(false), &Config::default());
        let config = app.compaction_config();
        // Config should be valid (just checking it returns something)
        let _ = config.enabled;
    }

    #[test]
    fn test_update_model_compaction_budget() {
        let mut app = App::new(test_options(false), &Config::default());
        app.model = "unknown-test-model".to_string();
        app.update_model_compaction_budget();
        let initial_threshold = app.compact_threshold;
        app.model = "deepseek-v3.2-128k".to_string();
        app.update_model_compaction_budget();
        // Threshold may have changed based on model
        // Explicit 128k DeepSeek model IDs have a higher threshold than unknown models.
        assert!(app.compact_threshold >= initial_threshold);
    }

    #[test]
    fn test_input_history_navigation() {
        let mut app = App::new(test_options(false), &Config::default());
        app.input_history.push("first".to_string());
        app.input_history.push("second".to_string());

        // Navigate up
        app.history_up();
        assert!(app.history_index.is_some());

        // Navigate down
        app.history_down();
    }

    #[test]
    fn input_history_down_restores_live_draft_after_accidental_up() {
        let mut app = App::new(test_options(false), &Config::default());
        app.input_history.push("previous prompt".to_string());
        app.input = "careful current draft".to_string();
        app.cursor_position = "careful".chars().count();

        app.history_up();
        assert_eq!(app.input, "previous prompt");

        app.history_down();
        assert_eq!(app.input, "careful current draft");
        assert_eq!(app.cursor_position, "careful".chars().count());
        assert!(app.history_index.is_none());
    }

    #[test]
    fn input_history_restores_empty_draft_at_end_of_navigation() {
        let mut app = App::new(test_options(false), &Config::default());
        app.input_history.push("previous prompt".to_string());

        app.history_up();
        assert_eq!(app.input, "previous prompt");

        app.history_down();
        assert!(app.input.is_empty());
        assert_eq!(app.cursor_position, 0);
        assert!(app.history_index.is_none());
    }

    #[test]
    fn word_cursor_helpers_move_by_whitespace_delimited_words() {
        let mut app = App::new(test_options(false), &Config::default());
        app.input = "alpha beta  gamma".to_string();
        app.cursor_position = 0;

        app.move_cursor_word_forward();
        assert_eq!(app.cursor_position, "alpha ".chars().count());

        app.move_cursor_word_forward();
        assert_eq!(app.cursor_position, "alpha beta  ".chars().count());

        app.move_cursor_word_backward();
        assert_eq!(app.cursor_position, "alpha ".chars().count());
    }

    #[test]
    fn editing_history_entry_leaves_navigation_mode() {
        let mut app = App::new(test_options(false), &Config::default());
        app.input_history.push("previous prompt".to_string());
        app.input = "current draft".to_string();
        app.cursor_position = app.input.chars().count();

        app.history_up();
        app.insert_char('!');
        app.history_down();

        assert_eq!(app.input, "previous prompt!");
        assert!(app.history_index.is_none());
    }

    #[test]
    fn history_search_filters_matches_and_skips_duplicates() {
        let mut app = App::new(test_options(false), &Config::default());
        app.input_history.clear();
        app.input_history.push("alpha one".to_string());
        app.input_history.push("beta two".to_string());
        app.input_history.push("alpha one".to_string());
        app.draft_history.push_back("draft alpha".to_string());

        app.start_history_search();
        app.history_search_insert_str("alpha");

        assert_eq!(
            app.history_search_matches(),
            vec!["draft alpha".to_string(), "alpha one".to_string()]
        );
    }

    #[test]
    fn history_search_matches_unicode_case_insensitively() {
        let mut app = App::new(test_options(false), &Config::default());
        app.input_history.clear();
        app.input_history.push("CAFÉ prompt".to_string());

        app.start_history_search();
        app.history_search_insert_str("café");

        assert_eq!(
            app.history_search_matches(),
            vec!["CAFÉ prompt".to_string()]
        );
    }

    #[test]
    fn history_search_accepts_match_without_submitting() {
        let mut app = App::new(test_options(false), &Config::default());
        app.input_history.clear();
        app.input_history.push("older prompt".to_string());

        app.start_history_search();
        app.history_search_insert_str("older");

        assert!(app.accept_history_search());
        assert_eq!(app.input, "older prompt");
        assert_eq!(app.cursor_position, "older prompt".chars().count());
        assert!(app.composer_history_search.is_none());
    }

    #[test]
    fn history_search_cancel_restores_pre_search_draft() {
        let mut app = App::new(test_options(false), &Config::default());
        app.input_history.clear();
        app.input = "current draft".to_string();
        app.cursor_position = 7;
        app.input_history.push("older prompt".to_string());

        app.start_history_search();
        app.history_search_insert_str("older");
        app.cancel_history_search();

        assert_eq!(app.input, "current draft");
        assert_eq!(app.cursor_position, 7);
        assert!(app.composer_history_search.is_none());
    }

    #[test]
    fn recoverable_clear_stashes_nonempty_draft() {
        let mut app = App::new(test_options(false), &Config::default());
        app.input_history.clear();
        app.input = "recover this".to_string();
        app.cursor_position = app.input.chars().count();

        app.clear_input_recoverable();
        app.start_history_search();
        app.history_search_insert_str("recover");

        assert_eq!(
            app.history_search_matches(),
            vec!["recover this".to_string()]
        );
    }

    #[test]
    fn composer_paste_flushes_pending_burst_and_normalizes_crlf() {
        let mut app = App::new(test_options(false), &Config::default());
        app.use_paste_burst_detection = true;
        let now = Instant::now();
        let key = crossterm::event::KeyEvent::new(
            crossterm::event::KeyCode::Char('x'),
            crossterm::event::KeyModifiers::NONE,
        );

        assert!(crate::tui::paste::handle_paste_burst_key(
            &mut app, &key, now
        ));
        assert!(
            app.input.is_empty(),
            "first burst char should stay buffered"
        );

        app.insert_paste_text("a\r\nb\rc");

        assert_eq!(app.input, "xa\nbc");
        assert_eq!(app.cursor_position, "xa\nbc".chars().count());
        assert!(!app.paste_burst.is_active());
    }

    #[test]
    fn enter_during_active_paste_burst_appends_newline_to_buffer_not_submit() {
        // #1073: when chars are still being assembled into a paste burst and
        // an Enter arrives (the trailing newline of the paste), the Enter
        // must be absorbed into the burst buffer — not fired as a submit.
        let mut app = App::new(test_options(false), &Config::default());
        app.use_paste_burst_detection = true;
        let now = Instant::now();
        app.paste_burst.append_char_to_buffer('h', now);
        app.paste_burst.append_char_to_buffer('i', now);
        assert!(app.paste_burst.is_active());
        assert!(app.input.is_empty());

        let result = app.handle_composer_enter();

        assert!(
            result.is_none(),
            "Enter during active paste burst must not submit"
        );
        let flushed = app.paste_burst.flush_before_modified_input();
        assert_eq!(
            flushed.as_deref(),
            Some("hi\n"),
            "newline must land in the burst buffer so the next flush carries it"
        );
    }

    #[test]
    fn enter_inside_paste_burst_window_after_flush_inserts_newline_not_submit() {
        // #1073: after a burst has flushed (text now in `input`), the
        // suppression window stays open for ~120ms. An Enter arriving in
        // that window is the trailing newline of the paste, not a user
        // submit — insert it as a literal newline into the composer.
        let mut app = App::new(test_options(false), &Config::default());
        app.use_paste_burst_detection = true;
        app.input = "hello".to_string();
        app.cursor_position = "hello".chars().count();
        let now = Instant::now();
        app.paste_burst.extend_window(now);
        assert!(!app.paste_burst.is_active());
        assert!(
            app.paste_burst.newline_should_insert_instead_of_submit(now),
            "suppression window should be open"
        );

        let result = app.handle_composer_enter();

        assert!(
            result.is_none(),
            "Enter inside post-flush suppression window must not submit"
        );
        assert_eq!(
            app.input, "hello\n",
            "newline must be inserted into the composer instead of firing a submit"
        );
    }

    #[test]
    fn enter_outside_any_paste_burst_window_submits_normally() {
        // Regression guard: the suppression must not trip when the user
        // actually wants to submit.
        let mut app = App::new(test_options(false), &Config::default());
        app.use_paste_burst_detection = true;
        app.input = "hello world".to_string();
        app.cursor_position = "hello world".chars().count();

        let result = app.handle_composer_enter();

        assert_eq!(
            result.as_deref(),
            Some("hello world"),
            "Enter outside any paste burst window must submit normally"
        );
        assert!(
            app.input.is_empty(),
            "submit_input should clear the composer"
        );
    }

    #[test]
    fn enter_with_paste_burst_detection_disabled_submits_normally() {
        // When the user has explicitly turned off paste-burst detection
        // (`bracketed_paste = false` is independent, this is the
        // `paste_burst_detection` setting), the suppression must be
        // skipped — otherwise turning it off would not actually turn it
        // off.
        let mut app = App::new(test_options(false), &Config::default());
        app.use_paste_burst_detection = false;
        app.input = "ship it".to_string();
        app.cursor_position = "ship it".chars().count();
        let now = Instant::now();
        app.paste_burst.extend_window(now);

        let result = app.handle_composer_enter();

        assert_eq!(result.as_deref(), Some("ship it"));
    }

    #[test]
    fn clipboard_text_paste_matches_bracketed_paste_state() {
        let text = "alpha\r\nbeta";
        let mut bracketed = App::new(test_options(false), &Config::default());
        let mut clipboard = App::new(test_options(false), &Config::default());

        bracketed.insert_paste_text(text);
        clipboard.apply_clipboard_content(ClipboardContent::Text(text.to_string()));

        assert_eq!(clipboard.input, bracketed.input);
        assert_eq!(clipboard.cursor_position, bracketed.cursor_position);
        assert_eq!(clipboard.slash_menu_hidden, bracketed.slash_menu_hidden);
        assert_eq!(clipboard.mention_menu_hidden, bracketed.mention_menu_hidden);
    }

    #[test]
    fn clipboard_image_paste_keeps_adjacent_text_and_concise_status() {
        let mut app = App::new(test_options(false), &Config::default());
        app.input = "before after".to_string();
        app.cursor_position = "before".chars().count();

        app.apply_clipboard_content(ClipboardContent::Image(PastedImage {
            path: PathBuf::from("/tmp/pasted.png"),
            width: 8,
            height: 4,
            byte_len: 2048,
        }));

        assert!(
            app.input
                .contains("before\n[Attached image: 8x4 PNG (2KB) at /tmp/pasted.png]")
        );
        assert!(app.input.contains("] after"));
        let status = app.status_message.as_deref().expect("status message");
        assert_eq!(status, "Attached image: 8x4 PNG (2KB)");
    }

    #[test]
    fn pasted_text_and_image_placeholders_survive_history_and_queue_paths() {
        let mut app = App::new(test_options(false), &Config::default());
        app.insert_paste_text("line 1\r\nline 2");
        app.insert_media_attachment("image", Path::new("/tmp/pasted.png"), Some("8x4 PNG (2KB)"));

        let submitted = app.submit_input().expect("submitted input");
        assert!(submitted.contains("line 1\nline 2"));
        assert!(submitted.contains("[Attached image: 8x4 PNG (2KB) at /tmp/pasted.png]"));

        app.history_up();
        assert_eq!(app.input, submitted);
        assert_eq!(app.composer_attachment_count(), 1);

        app.clear_input();
        app.queue_message(QueuedMessage::new(
            submitted.clone(),
            Some("Use this skill".to_string()),
        ));
        assert!(app.pop_last_queued_into_draft());
        assert_eq!(app.input, submitted);
        assert_eq!(app.composer_attachment_count(), 1);
        assert_eq!(
            app.queued_draft
                .as_ref()
                .and_then(|draft| draft.skill_instruction.as_deref()),
            Some("Use this skill")
        );

        app.push_pending_steer(QueuedMessage::new(submitted.clone(), None));
        let steers = app.drain_pending_steers();
        assert_eq!(steers[0].display, submitted);
    }

    #[test]
    fn selected_attachment_row_removes_placeholder_without_manual_editing() {
        let mut app = App::new(test_options(false), &Config::default());
        app.input = "before".to_string();
        app.cursor_position = "before".chars().count();
        app.insert_media_attachment("image", Path::new("/tmp/pasted.png"), Some("8x4 PNG"));
        app.insert_str("after");

        app.move_cursor_start();
        assert!(app.select_previous_composer_attachment());
        assert_eq!(app.selected_composer_attachment_index(), Some(0));
        assert!(app.remove_selected_composer_attachment());

        assert!(!app.input.contains("[Attached image:"));
        assert!(app.input.contains("before"));
        assert!(app.input.contains("after"));
        assert_eq!(app.composer_attachment_count(), 0);
        assert!(app.selected_composer_attachment_index().is_none());
    }

    #[test]
    fn kill_to_end_of_line_cuts_from_middle_of_word() {
        let mut app = App::new(test_options(false), &Config::default());
        app.input = "hello world".to_string();
        app.cursor_position = 6; // before 'w'
        assert!(app.kill_to_end_of_line());
        assert_eq!(app.input, "hello ");
        assert_eq!(app.cursor_position, 6);
        assert_eq!(app.kill_buffer, "world");
    }

    #[test]
    fn kill_at_eol_consumes_following_newline() {
        let mut app = App::new(test_options(false), &Config::default());
        app.input = "line one\nline two".to_string();
        app.cursor_position = 8; // sitting on the '\n'
        assert!(app.kill_to_end_of_line());
        assert_eq!(app.input, "line oneline two");
        assert_eq!(app.cursor_position, 8);
        assert_eq!(app.kill_buffer, "\n");

        // Empty input: kill is a no-op and the buffer is untouched.
        let mut empty = App::new(test_options(false), &Config::default());
        assert!(!empty.kill_to_end_of_line());
        assert!(empty.input.is_empty());
        assert!(empty.kill_buffer.is_empty());
    }

    #[test]
    fn yank_inserts_kill_buffer_and_preserves_it() {
        let mut app = App::new(test_options(false), &Config::default());
        app.input = "abc def".to_string();
        app.cursor_position = 4; // before 'd'
        assert!(app.kill_to_end_of_line());
        assert_eq!(app.input, "abc ");
        assert_eq!(app.kill_buffer, "def");

        // Move cursor to the start and yank twice — kill_buffer must persist.
        app.cursor_position = 0;
        assert!(app.yank());
        assert!(app.yank());
        assert_eq!(app.input, "defdefabc ");
        assert_eq!(app.cursor_position, 6);
        assert_eq!(app.kill_buffer, "def");

        // Yank with empty buffer is a no-op.
        let mut empty = App::new(test_options(false), &Config::default());
        assert!(!empty.yank());
        assert!(empty.input.is_empty());
    }

    // ---- Issue #90: quit confirmation timeout ----

    #[test]
    fn quit_is_not_armed_by_default() {
        let app = App::new(test_options(false), &Config::default());
        assert!(!app.quit_is_armed());
        assert!(app.quit_armed_until.is_none());
    }

    #[test]
    fn arm_quit_sets_two_second_window() {
        let mut app = App::new(test_options(false), &Config::default());
        app.arm_quit();
        assert!(app.quit_is_armed());
        let deadline = app.quit_armed_until.expect("deadline set");
        let remaining = deadline.saturating_duration_since(Instant::now());
        // Allow a generous margin for slow CI machines: 1.5s..=2.0s.
        assert!(
            remaining >= Duration::from_millis(1500) && remaining <= Duration::from_secs(2),
            "expected ~2s window, got {remaining:?}",
        );
        assert!(app.needs_redraw, "armed prompt should request a redraw");
    }

    #[test]
    fn disarm_quit_clears_the_timer() {
        let mut app = App::new(test_options(false), &Config::default());
        app.arm_quit();
        app.needs_redraw = false;
        app.disarm_quit();
        assert!(!app.quit_is_armed());
        assert!(app.quit_armed_until.is_none());
        assert!(app.needs_redraw, "disarming should request a redraw");
    }

    #[test]
    fn disarm_quit_when_not_armed_is_a_noop() {
        let mut app = App::new(test_options(false), &Config::default());
        app.needs_redraw = false;
        app.disarm_quit();
        assert!(!app.needs_redraw, "no redraw when nothing changed");
    }

    #[test]
    fn quit_armed_expires_after_window() {
        let mut app = App::new(test_options(false), &Config::default());
        // Pin the deadline in the past to simulate a stale timer.
        app.quit_armed_until = Some(Instant::now() - Duration::from_millis(10));
        assert!(
            !app.quit_is_armed(),
            "expired timer must not count as armed"
        );

        app.needs_redraw = false;
        app.tick_quit_armed();
        assert!(app.quit_armed_until.is_none(), "tick clears expired timer");
        assert!(
            app.needs_redraw,
            "expiry triggers a redraw to repaint footer"
        );
    }

    #[test]
    fn quit_armed_tick_is_noop_within_window() {
        let mut app = App::new(test_options(false), &Config::default());
        app.arm_quit();
        app.needs_redraw = false;
        app.tick_quit_armed();
        assert!(
            app.quit_is_armed(),
            "tick within window keeps the timer armed"
        );
        assert!(!app.needs_redraw, "no redraw when nothing changed");
    }

    #[test]
    fn re_arming_after_expiry_starts_a_fresh_window() {
        let mut app = App::new(test_options(false), &Config::default());
        app.quit_armed_until = Some(Instant::now() - Duration::from_secs(5));
        app.tick_quit_armed();
        assert!(app.quit_armed_until.is_none());
        app.arm_quit();
        let deadline = app.quit_armed_until.expect("re-armed");
        assert!(deadline > Instant::now(), "fresh deadline in the future");
    }

    // ---- Issue #208: in-flight input routing ----

    #[test]
    fn submit_disposition_immediate_when_idle_and_online() {
        let app = App::new(test_options(false), &Config::default());
        assert!(!app.is_loading);
        assert!(!app.offline_mode);
        assert_eq!(
            app.decide_submit_disposition(),
            SubmitDisposition::Immediate
        );
    }

    #[test]
    fn submit_disposition_queue_when_busy_and_online_not_streaming() {
        // #382: Busy + not streaming → Queue (was Steer; now unified)
        let mut app = App::new(test_options(false), &Config::default());
        app.is_loading = true;
        app.offline_mode = false;
        // streaming_message_index is None (default) → tool execution phase
        assert_eq!(app.decide_submit_disposition(), SubmitDisposition::Queue);
    }

    #[test]
    fn submit_disposition_queue_when_busy_and_streaming() {
        // #382: Busy + streaming → Queue (was QueueFollowUp; now unified)
        let mut app = App::new(test_options(false), &Config::default());
        app.is_loading = true;
        app.offline_mode = false;
        app.streaming_message_index = Some(0);
        assert_eq!(app.decide_submit_disposition(), SubmitDisposition::Queue);
    }

    #[test]
    fn submit_disposition_queue_when_offline_and_idle() {
        let mut app = App::new(test_options(false), &Config::default());
        app.is_loading = false;
        app.offline_mode = true;
        assert_eq!(app.decide_submit_disposition(), SubmitDisposition::Queue);
    }

    #[test]
    fn submit_disposition_offline_busy_queues() {
        let mut app = App::new(test_options(false), &Config::default());
        app.is_loading = true;
        app.offline_mode = true;
        // Offline mode always queues, even when streaming
        app.streaming_message_index = Some(0);
        assert_eq!(app.decide_submit_disposition(), SubmitDisposition::Queue);
    }

    #[test]
    fn push_pending_steer_arms_resend_flag() {
        let mut app = App::new(test_options(false), &Config::default());
        assert!(!app.submit_pending_steers_after_interrupt);
        app.push_pending_steer(QueuedMessage::new("steer me".to_string(), None));
        assert_eq!(app.pending_steers.len(), 1);
        assert!(app.submit_pending_steers_after_interrupt);
    }

    #[test]
    fn drain_pending_steers_clears_flag_and_returns_in_order() {
        let mut app = App::new(test_options(false), &Config::default());
        app.push_pending_steer(QueuedMessage::new("first".to_string(), None));
        app.push_pending_steer(QueuedMessage::new("second".to_string(), None));
        app.push_pending_steer(QueuedMessage::new("third".to_string(), None));

        let drained = app.drain_pending_steers();
        assert_eq!(drained.len(), 3);
        assert_eq!(drained[0].display, "first");
        assert_eq!(drained[2].display, "third");
        assert!(app.pending_steers.is_empty());
        assert!(!app.submit_pending_steers_after_interrupt);
    }

    #[test]
    fn drain_pending_steers_when_empty_is_safe() {
        let mut app = App::new(test_options(false), &Config::default());
        // Flag-only set (someone armed it manually): drain still clears it.
        app.submit_pending_steers_after_interrupt = true;
        let drained = app.drain_pending_steers();
        assert!(drained.is_empty());
        assert!(!app.submit_pending_steers_after_interrupt);
    }

    #[test]
    fn double_push_pending_steer_is_idempotent_on_flag() {
        let mut app = App::new(test_options(false), &Config::default());
        app.push_pending_steer(QueuedMessage::new("a".to_string(), None));
        app.push_pending_steer(QueuedMessage::new("b".to_string(), None));
        assert!(app.submit_pending_steers_after_interrupt);
        assert_eq!(app.pending_steers.len(), 2);
    }

    #[test]
    fn pop_last_queued_into_draft_pops_back_and_arms_draft() {
        let mut app = App::new(test_options(false), &Config::default());
        app.queue_message(QueuedMessage::new(
            "first".to_string(),
            Some("skill-A".to_string()),
        ));
        app.queue_message(QueuedMessage::new(
            "last".to_string(),
            Some("skill-B".to_string()),
        ));

        assert!(app.pop_last_queued_into_draft());
        assert_eq!(app.input, "last");
        assert_eq!(app.cursor_position, "last".chars().count());
        assert_eq!(app.queued_messages.len(), 1);
        let draft = app.queued_draft.clone().expect("draft is set");
        assert_eq!(draft.display, "last");
        assert_eq!(draft.skill_instruction.as_deref(), Some("skill-B"));
    }

    #[test]
    fn pop_last_queued_into_draft_noop_when_composer_dirty() {
        let mut app = App::new(test_options(false), &Config::default());
        app.queue_message(QueuedMessage::new("queued".to_string(), None));
        app.input = "typing".to_string();
        app.cursor_position = char_count(&app.input);

        assert!(!app.pop_last_queued_into_draft());
        assert_eq!(app.input, "typing");
        assert_eq!(app.queued_messages.len(), 1);
        assert!(app.queued_draft.is_none());
    }

    #[test]
    fn pop_last_queued_into_draft_noop_when_draft_already_armed() {
        let mut app = App::new(test_options(false), &Config::default());
        app.queue_message(QueuedMessage::new("queued".to_string(), None));
        app.queued_draft = Some(QueuedMessage::new("editing".to_string(), None));

        assert!(!app.pop_last_queued_into_draft());
        assert_eq!(app.queued_messages.len(), 1);
        assert_eq!(
            app.queued_draft.as_ref().map(|d| d.display.as_str()),
            Some("editing")
        );
    }

    #[test]
    fn pop_last_queued_into_draft_noop_when_queue_empty() {
        let mut app = App::new(test_options(false), &Config::default());
        assert!(!app.pop_last_queued_into_draft());
        assert!(app.input.is_empty());
        assert!(app.queued_draft.is_none());
    }

    #[test]
    fn finalize_streaming_assistant_marks_existing_cell_interrupted() {
        let mut app = App::new(test_options(false), &Config::default());
        app.add_message(HistoryCell::Assistant {
            content: "partial reply so far".to_string(),
            streaming: true,
        });
        let idx = app.history.len() - 1;
        app.streaming_message_index = Some(idx);

        app.finalize_streaming_assistant_as_interrupted();

        assert!(app.streaming_message_index.is_none());
        match &app.history[idx] {
            HistoryCell::Assistant { content, streaming } => {
                assert!(content.starts_with("[interrupted]"), "got: {content}");
                assert!(content.contains("partial reply so far"));
                assert!(!streaming);
            }
            other => panic!("expected Assistant cell, got {other:?}"),
        }
    }

    #[test]
    fn finalize_streaming_assistant_handles_empty_content() {
        let mut app = App::new(test_options(false), &Config::default());
        app.add_message(HistoryCell::Assistant {
            content: String::new(),
            streaming: true,
        });
        let idx = app.history.len() - 1;
        app.streaming_message_index = Some(idx);

        app.finalize_streaming_assistant_as_interrupted();

        match &app.history[idx] {
            HistoryCell::Assistant { content, streaming } => {
                assert_eq!(content, "[interrupted]");
                assert!(!streaming);
            }
            other => panic!("expected Assistant cell, got {other:?}"),
        }
    }

    #[test]
    fn finalize_streaming_assistant_no_op_without_index() {
        let mut app = App::new(test_options(false), &Config::default());
        // No streaming index set; should not panic and should leave history unchanged.
        let prev_len = app.history.len();
        app.finalize_streaming_assistant_as_interrupted();
        assert_eq!(app.history.len(), prev_len);
        assert!(app.streaming_message_index.is_none());
    }

    #[test]
    fn finalize_streaming_assistant_is_idempotent_on_double_call() {
        let mut app = App::new(test_options(false), &Config::default());
        app.add_message(HistoryCell::Assistant {
            content: "something".to_string(),
            streaming: true,
        });
        let idx = app.history.len() - 1;
        app.streaming_message_index = Some(idx);

        app.finalize_streaming_assistant_as_interrupted();
        // Second call without resetting state must be safe.
        app.finalize_streaming_assistant_as_interrupted();

        match &app.history[idx] {
            HistoryCell::Assistant { content, .. } => {
                // Second call still finds index None — content unchanged from first.
                assert!(content.starts_with("[interrupted] "));
                assert_eq!(content.matches("[interrupted]").count(), 1);
            }
            other => panic!("expected Assistant cell, got {other:?}"),
        }
    }

    #[test]
    fn delete_word_backward_removes_previous_word_only() {
        let mut app = App::new(test_options(false), &Config::default());
        app.input = "hello world".to_string();
        app.cursor_position = char_count(&app.input);

        app.delete_word_backward();

        assert_eq!(app.input, "hello ");
        assert_eq!(app.cursor_position, char_count("hello "));
    }

    #[test]
    fn delete_word_backward_handles_trailing_space_and_utf8() {
        let mut app = App::new(test_options(false), &Config::default());
        app.input = "cafe 你好   ".to_string();
        app.cursor_position = char_count(&app.input);

        app.delete_word_backward();

        assert_eq!(app.input, "cafe ");
        assert_eq!(app.cursor_position, char_count("cafe "));
    }

    #[test]
    fn delete_word_forward_handles_leading_space_and_utf8() {
        let mut app = App::new(test_options(false), &Config::default());
        app.input = "hello 你好 world".to_string();
        app.cursor_position = char_count("hello");

        app.delete_word_forward();

        assert_eq!(app.input, "hello world");
        assert_eq!(app.cursor_position, char_count("hello"));
    }

    #[test]
    fn delete_to_start_of_line_respects_multiline_cursor() {
        let mut app = App::new(test_options(false), &Config::default());
        app.input = "first\nsecond line".to_string();
        app.cursor_position = char_count("first\nsecond");

        app.delete_to_start_of_line();

        assert_eq!(app.input, "first\n line");
        assert_eq!(app.cursor_position, char_count("first\n"));
    }

    #[test]
    fn kill_and_yank_handle_multibyte_utf8() {
        let mut app = App::new(test_options(false), &Config::default());
        // "café 你好" — char_count = 7 (c,a,f,é, ,你,好); UTF-8 bytes differ.
        app.input = "café 你好".to_string();
        app.cursor_position = 5; // before '你'
        assert!(app.kill_to_end_of_line());
        assert_eq!(app.input, "café ");
        assert_eq!(app.cursor_position, 5);
        assert_eq!(app.kill_buffer, "你好");

        // Yank back at the same spot — must not panic on char boundaries.
        assert!(app.yank());
        assert_eq!(app.input, "café 你好");
        assert_eq!(app.cursor_position, 7);
    }
}
