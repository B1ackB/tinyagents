//! [`VerifyBeforeFinishMiddleware`]: before a multi-step run's first final
//! answer stands, ask the model once to check it against the request.
//!
//! # Why
//!
//! A run ends the moment the model returns text with no tool calls, i.e. when
//! the model *believes* it is done. On long tasks that belief is often built on
//! checks that cannot fail: a test derived from the implementation rather than
//! the spec, a benchmark measuring something other than what is graded, a
//! filter stated in the request that nothing ever re-read. One extra model call
//! that re-reads the request against the finished work is the cheapest lever
//! the harness has on that failure mode. This is a hypothesis about model
//! behaviour, so the middleware is opt-in and the trigger is the host's call.
//!
//! # How
//!
//! `after_model` sees the response before the loop decides it is final. When
//! it is a final answer (text, no tool calls, not truncated, not already
//! continued), the run's activity meets the trigger, the budget has room and
//! the check has not run yet in this run, the middleware sets
//! [`ModelResponse::continue_turn`] to the check message. The loop then keeps
//! the draft answer on the transcript, appends the check as the next **user**
//! turn (tail content, never a mid-conversation system message, so the cached
//! prefix is untouched) and asks for another reply. That reply is free to call
//! tools and fix what the check found; the next tool-less answer ends the run,
//! because the check fires at most once per run.
//!
//! The check rides `continue_turn` rather than a `JumpTo(Model)` plus a
//! `before_model` injection so the message is part of the transcript: a
//! follow-up call that fixes something still sees the instruction it is acting
//! on, and every later request keeps the same prefix.
//!
//! # Budget
//!
//! Skipped when two or fewer model calls remain after the answer, so it never
//! competes with a final-call wrap-up for the last calls, and when a configured
//! wall-clock deadline is closer than
//! [`with_min_remaining_wall_clock`](VerifyBeforeFinishMiddleware::with_min_remaining_wall_clock)
//! (the run context's deadline, and the policy cap a host declares through
//! [`with_wall_clock_limit`](VerifyBeforeFinishMiddleware::with_wall_clock_limit)).

mod middleware;
mod types;

pub use types::{
    DEFAULT_MIN_REMAINING_WALL_CLOCK, FinishActivity, FinishCheckTrigger,
    MIN_REMAINING_MODEL_CALLS, VerifyBeforeFinishMiddleware,
};
