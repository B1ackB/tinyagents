//! Host seams for the configurable hook engine.
//!
//! Everything the engine needs from its embedder that is not part of the
//! `hooks.json` contract itself lives here: the product name that decides where
//! system-wide and per-user hook files are discovered and what the hook
//! process environment variables are called, the user's home directory, the
//! shell used to run `command` hooks, and the model evaluator behind `prompt`
//! hooks.

use std::path::PathBuf;
use std::sync::Arc;

use async_trait::async_trait;

/// Builds the process a `command` hook runs in, from the hook's command line.
///
/// The engine sets the working directory, environment and stdio itself; the
/// factory only chooses the shell (`bash -lc`, `cmd /C`, ...).
pub type ShellFactory = fn(&str) -> tokio::process::Command;

/// Evaluates a `prompt` hook: sends an instruction to a model and returns its
/// raw answer. Errors are plain strings because the only consumer folds them
/// into the fail-open / fail-closed decision.
#[async_trait]
pub trait PromptEvaluator: Send + Sync {
    /// Ask the model to judge `instruction`, optionally overriding the model.
    async fn evaluate(&self, instruction: &str, model: Option<&str>) -> Result<String, String>;
}

/// Everything the hook engine takes from its host.
#[derive(Clone)]
pub struct HookEnvironment {
    /// Product name, in display case (for example `OpenHuman`).
    ///
    /// Derives the hook file locations: `%ProgramData%\<product>` on Windows,
    /// `/Library/Application Support/<product>` on macOS,
    /// `/etc/<product lowercase>` elsewhere, and the `.<product lowercase>`
    /// directory in the user's home and in a project. It also derives the
    /// `<PRODUCT UPPERCASE>_*` variables passed to hook processes.
    pub product: String,
    /// The user's home directory, or `None` to skip the user layer.
    pub home_dir: Option<PathBuf>,
    /// Shell used to run `command` hooks.
    pub shell: ShellFactory,
    /// Evaluator for `prompt` hooks; without one they fail like any other
    /// hook that could not run.
    pub prompt_evaluator: Option<Arc<dyn PromptEvaluator>>,
}

impl HookEnvironment {
    /// An environment for `product` with the default shell and no prompt
    /// evaluator.
    pub fn new(product: impl Into<String>, home_dir: Option<PathBuf>) -> Self {
        Self {
            product: product.into(),
            home_dir,
            shell: default_shell,
            prompt_evaluator: None,
        }
    }

    /// Replace the shell factory.
    pub fn with_shell(mut self, shell: ShellFactory) -> Self {
        self.shell = shell;
        self
    }

    /// Install the `prompt` hook evaluator.
    pub fn with_prompt_evaluator(mut self, evaluator: Arc<dyn PromptEvaluator>) -> Self {
        self.prompt_evaluator = Some(evaluator);
        self
    }

    /// `.openhuman`-style directory name.
    pub(crate) fn dot_dir(&self) -> String {
        format!(".{}", self.product.to_lowercase())
    }

    /// Prefix of the environment variables handed to hook processes.
    pub(crate) fn env_prefix(&self) -> String {
        self.product.to_uppercase()
    }
}

impl Default for HookEnvironment {
    fn default() -> Self {
        Self::new("tinyagents", None)
    }
}

impl std::fmt::Debug for HookEnvironment {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HookEnvironment")
            .field("product", &self.product)
            .field("home_dir", &self.home_dir)
            .field("prompt_evaluator", &self.prompt_evaluator.is_some())
            .finish()
    }
}

/// `sh -c` (`cmd /C` on Windows): the shell used when the host names none.
pub fn default_shell(command: &str) -> tokio::process::Command {
    #[cfg(windows)]
    {
        let mut cmd = tokio::process::Command::new("cmd");
        cmd.arg("/C").arg(command);
        cmd
    }
    #[cfg(not(windows))]
    {
        let mut cmd = tokio::process::Command::new("sh");
        cmd.arg("-c").arg(command);
        cmd
    }
}
