//! Turn clock stub (#6953).

use std::time::Duration;

use async_trait::async_trait;

use crate::context::RunContext;
use crate::middleware::Middleware;

pub(crate) fn format_clock(_d: Duration) -> String {
    String::new()
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TurnClock {
    pub elapsed: Duration,
    pub budget: Duration,
}

impl TurnClock {
    pub fn of<C>(_ctx: &RunContext<C>, _host_budget: Option<Duration>) -> Option<Self> {
        None
    }
    pub fn remaining(&self) -> Duration {
        Duration::ZERO
    }
    pub fn band(&self) -> Option<u32> {
        None
    }
    pub fn note(&self) -> String {
        String::new()
    }
}

pub struct TurnClockMiddleware;

impl TurnClockMiddleware {
    pub fn new(_budget: Option<Duration>) -> Self {
        Self
    }
}

#[async_trait]
impl<S: Send + Sync, C: Send + Sync> Middleware<S, C> for TurnClockMiddleware {
    fn name(&self) -> &str {
        "turn_clock"
    }
}

#[cfg(test)]
#[path = "turn_clock_tests.rs"]
mod tests;
