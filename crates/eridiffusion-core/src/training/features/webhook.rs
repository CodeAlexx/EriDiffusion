//! Phase 7: Webhook notifications.
//!
//! Posts a small JSON payload to a Discord/Slack-compatible webhook URL.
//! Both Discord (`content`) and Slack (`text`) field names are included so
//! the same client works for either provider with no per-call branching.
//!
//! Sync HTTP via `ureq` — no tokio runtime, no async ceremony.

use serde_json::json;
use std::time::Duration;

pub struct WebhookClient {
    pub url: String,
    pub timeout: Duration,
}

impl WebhookClient {
    pub fn new(url: String) -> Self {
        Self {
            url,
            timeout: Duration::from_secs(5),
        }
    }

    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
    }

    /// Best-effort POST. Failures are logged at WARN and swallowed — webhook
    /// outages must NOT crash training. Discord-compatible JSON with a Slack
    /// `text` mirror so a single payload works for either provider.
    pub fn send(&self, message: &str) {
        let body = json!({
            "content": message,
            "text": message,
        });
        let agent = ureq::AgentBuilder::new().timeout(self.timeout).build();
        match agent.post(&self.url).send_json(body) {
            Ok(_) => log::debug!("[webhook] sent: {}", message),
            Err(e) => log::warn!("[webhook] send failed: {}", e),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn webhook_client_constructs() {
        let c = WebhookClient::new("https://example.invalid/webhook".to_string());
        assert_eq!(c.url, "https://example.invalid/webhook");
        assert_eq!(c.timeout, Duration::from_secs(5));
    }

    #[test]
    fn webhook_send_failure_does_not_panic() {
        // `.invalid` TLD is reserved + never resolves — call must swallow
        // the error and return cleanly.
        let c = WebhookClient::new("https://nonexistent.invalid/x".to_string())
            .with_timeout(Duration::from_millis(100));
        c.send("test message");
    }

    #[test]
    fn webhook_with_timeout_overrides_default() {
        let c = WebhookClient::new("https://example.invalid/webhook".to_string())
            .with_timeout(Duration::from_secs(30));
        assert_eq!(c.timeout, Duration::from_secs(30));
    }
}
