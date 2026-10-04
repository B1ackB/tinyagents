use tinyinference_llm::message::Message;

/// Immutable messages which remain at the front of a session's history.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct PrefixSnapshot {
    messages: Vec<Message>,
    refresh: bool,
}

impl PrefixSnapshot {
    /// Captures the prefix once, before turn history starts growing.
    pub fn new(messages: Vec<Message>) -> Self {
        Self {
            messages,
            refresh: false,
        }
    }

    /// Explicitly allow replacing this prefix after committed turns.
    ///
    /// The runtime preserves conversation rows and writes a successor transcript
    /// generation when the prefix changes. Ordinary snapshots remain frozen.
    pub fn refreshing(mut self) -> Self {
        self.refresh = true;
        self
    }

    pub(crate) fn allows_refresh(&self) -> bool {
        self.refresh
    }
    pub(crate) fn frozen(mut self) -> Self {
        self.refresh = false;
        self
    }

    /// Returns the captured messages in their original order.
    pub fn messages(&self) -> &[Message] {
        &self.messages
    }
}
