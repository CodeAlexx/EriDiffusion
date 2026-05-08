//! SerenityBoard SQLite metrics writer.
//!
//! Schema (matches /home/alex/serenityflow-v2/serenityflow/debug/training_reader.py):
//!
//! ```sql
//! CREATE TABLE sessions (
//!   session_id TEXT PRIMARY KEY,
//!   start_time REAL NOT NULL,
//!   resume_step INTEGER,
//!   status TEXT NOT NULL
//! ) WITHOUT ROWID;
//!
//! CREATE TABLE scalars (
//!   tag TEXT NOT NULL,
//!   step INTEGER NOT NULL,
//!   wall_time REAL NOT NULL,
//!   value REAL NOT NULL,
//!   PRIMARY KEY (tag, step)
//! ) WITHOUT ROWID;
//! ```
//!
//! Output convention: `<output_dir>/board.db`. SerenityBoard's UI scans
//! `log_dir/*/board.db` and picks the most recently modified one.

use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::time::{SystemTime, UNIX_EPOCH};

use rusqlite::{params, Connection};

pub struct BoardWriter {
    conn: Mutex<Connection>,
    session_id: String,
    pub db_path: PathBuf,
}

fn unix_now() -> f64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs_f64())
        .unwrap_or(0.0)
}

impl BoardWriter {
    /// Open or create `<output_dir>/board.db`, initialize schema, insert a
    /// `running` session row. `resume_step` is the step number we're resuming
    /// from (None = fresh run).
    pub fn open(
        output_dir: &Path,
        session_id: impl Into<String>,
        resume_step: Option<u64>,
    ) -> rusqlite::Result<Self> {
        std::fs::create_dir_all(output_dir).ok();
        let db_path = output_dir.join("board.db");
        let conn = Connection::open(&db_path)?;
        // Full SerenityBoard V4 schema. Mirrors
        // /home/alex/serenityboard/serenityboard/writer/schema.py so the
        // dashboard's data_provider can issue any of its standard queries
        // (tensors, artifacts, audio, trace_events, etc.) without 'no such
        // table' errors. We only write to scalars + sessions + metadata; the
        // rest stay empty but valid.
        conn.execute_batch(
            r#"
            PRAGMA journal_mode = WAL;
            PRAGMA synchronous = NORMAL;
            PRAGMA busy_timeout = 5000;
            CREATE TABLE IF NOT EXISTS metadata (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS sessions (
                session_id TEXT NOT NULL,
                start_time REAL NOT NULL,
                resume_step INTEGER,
                status TEXT NOT NULL CHECK(status IN ('running','complete','crashed')),
                PRIMARY KEY (session_id)
            ) WITHOUT ROWID;
            CREATE TABLE IF NOT EXISTS scalars (
                tag TEXT NOT NULL,
                step INTEGER NOT NULL,
                wall_time REAL NOT NULL,
                value REAL NOT NULL,
                PRIMARY KEY (tag, step)
            ) WITHOUT ROWID;
            CREATE TABLE IF NOT EXISTS tensors (
                tag TEXT NOT NULL, step INTEGER NOT NULL, wall_time REAL NOT NULL,
                dtype TEXT NOT NULL, shape TEXT NOT NULL, data BLOB NOT NULL,
                PRIMARY KEY (tag, step)
            ) WITHOUT ROWID;
            CREATE TABLE IF NOT EXISTS artifacts (
                tag TEXT NOT NULL, step INTEGER NOT NULL,
                seq_index INTEGER NOT NULL DEFAULT 0,
                wall_time REAL NOT NULL, kind TEXT NOT NULL,
                mime_type TEXT NOT NULL, blob_key TEXT NOT NULL,
                width INTEGER, height INTEGER,
                meta TEXT NOT NULL DEFAULT '{}',
                PRIMARY KEY (tag, step, seq_index)
            ) WITHOUT ROWID;
            CREATE TABLE IF NOT EXISTS text_events (
                tag TEXT NOT NULL, step INTEGER NOT NULL,
                wall_time REAL NOT NULL, value TEXT NOT NULL,
                PRIMARY KEY (tag, step)
            ) WITHOUT ROWID;
            CREATE TABLE IF NOT EXISTS trace_events (
                step INTEGER NOT NULL, wall_time REAL NOT NULL,
                phase TEXT NOT NULL, duration_ms REAL NOT NULL,
                details TEXT NOT NULL DEFAULT '{}',
                PRIMARY KEY (step, phase)
            ) WITHOUT ROWID;
            CREATE TABLE IF NOT EXISTS eval_results (
                suite_name TEXT NOT NULL, case_id TEXT NOT NULL,
                step INTEGER NOT NULL, wall_time REAL NOT NULL,
                score_name TEXT NOT NULL, score_value REAL NOT NULL,
                artifact_key TEXT, details TEXT NOT NULL DEFAULT '{}',
                PRIMARY KEY (suite_name, case_id, step, score_name)
            ) WITHOUT ROWID;
            CREATE TABLE IF NOT EXISTS hparam_metrics (
                metric_tag TEXT NOT NULL, value REAL NOT NULL,
                step INTEGER, wall_time REAL,
                PRIMARY KEY (metric_tag)
            ) WITHOUT ROWID;
            CREATE TABLE IF NOT EXISTS plugin_data (
                plugin_name TEXT NOT NULL, tag TEXT NOT NULL,
                step INTEGER NOT NULL, wall_time REAL NOT NULL,
                data TEXT NOT NULL,
                PRIMARY KEY (plugin_name, tag, step)
            ) WITHOUT ROWID;
            CREATE TABLE IF NOT EXISTS custom_scalar_layouts (
                layout_name TEXT NOT NULL, config TEXT NOT NULL,
                PRIMARY KEY (layout_name)
            ) WITHOUT ROWID;
            CREATE TABLE IF NOT EXISTS pr_curves (
                tag TEXT NOT NULL, step INTEGER NOT NULL,
                class_index INTEGER NOT NULL DEFAULT 0,
                wall_time REAL NOT NULL,
                num_thresholds INTEGER NOT NULL, data BLOB NOT NULL,
                PRIMARY KEY (tag, step, class_index)
            ) WITHOUT ROWID;
            CREATE TABLE IF NOT EXISTS audio (
                tag TEXT NOT NULL, step INTEGER NOT NULL,
                seq_index INTEGER NOT NULL DEFAULT 0,
                wall_time REAL NOT NULL, blob_key TEXT NOT NULL,
                sample_rate INTEGER NOT NULL,
                num_channels INTEGER NOT NULL DEFAULT 1,
                duration_ms REAL,
                mime_type TEXT NOT NULL DEFAULT 'audio/wav',
                label TEXT NOT NULL DEFAULT '',
                PRIMARY KEY (tag, step, seq_index)
            ) WITHOUT ROWID;
            CREATE TABLE IF NOT EXISTS graphs (
                tag TEXT NOT NULL, step INTEGER NOT NULL,
                wall_time REAL NOT NULL, graph_blob_key TEXT NOT NULL,
                PRIMARY KEY (tag, step)
            ) WITHOUT ROWID;
            CREATE TABLE IF NOT EXISTS embeddings (
                tag TEXT NOT NULL, step INTEGER NOT NULL,
                wall_time REAL NOT NULL,
                num_points INTEGER NOT NULL, dimensions INTEGER NOT NULL,
                tensor_blob_key TEXT NOT NULL,
                metadata_json TEXT, metadata_header TEXT,
                sprite_blob_key TEXT,
                sprite_single_h INTEGER, sprite_single_w INTEGER,
                PRIMARY KEY (tag, step)
            ) WITHOUT ROWID;
            CREATE TABLE IF NOT EXISTS meshes (
                tag TEXT NOT NULL, step INTEGER NOT NULL,
                wall_time REAL NOT NULL,
                num_vertices INTEGER NOT NULL,
                has_faces INTEGER NOT NULL DEFAULT 0,
                has_colors INTEGER NOT NULL DEFAULT 0,
                num_faces INTEGER NOT NULL DEFAULT 0,
                vertices_blob_key TEXT NOT NULL,
                faces_blob_key TEXT, colors_blob_key TEXT,
                config_json TEXT,
                PRIMARY KEY (tag, step)
            ) WITHOUT ROWID;
            CREATE INDEX IF NOT EXISTS idx_scalars_tag ON scalars(tag);
            CREATE INDEX IF NOT EXISTS idx_scalars_tag_step ON scalars(tag, step DESC);
            CREATE INDEX IF NOT EXISTS idx_tensors_tag_step ON tensors(tag, step DESC);
            CREATE INDEX IF NOT EXISTS idx_artifacts_tag_step ON artifacts(tag, step DESC);
            CREATE INDEX IF NOT EXISTS idx_text_tag_step ON text_events(tag, step DESC);
            CREATE INDEX IF NOT EXISTS idx_eval_suite_step ON eval_results(suite_name, step DESC);
            CREATE INDEX IF NOT EXISTS idx_plugin_name_tag ON plugin_data(plugin_name, tag);
            CREATE INDEX IF NOT EXISTS idx_pr_curves_tag_step ON pr_curves(tag, step DESC);
            CREATE INDEX IF NOT EXISTS idx_audio_tag_step ON audio(tag, step DESC);
            CREATE INDEX IF NOT EXISTS idx_graphs_tag_step ON graphs(tag, step DESC);
            CREATE INDEX IF NOT EXISTS idx_embeddings_tag_step ON embeddings(tag, step DESC);
            CREATE INDEX IF NOT EXISTS idx_meshes_tag_step ON meshes(tag, step DESC);
            "#,
        )?;

        let session_id: String = session_id.into();
        let now = unix_now();
        conn.execute(
            "INSERT OR REPLACE INTO sessions (session_id, start_time, resume_step, status) VALUES (?, ?, ?, 'running')",
            params![session_id, now, resume_step.map(|s| s as i64)],
        )?;

        // SerenityBoard's `data_provider.py` reads `active_session_id` (JSON-encoded)
        // from the metadata table to know which session is current. Other keys
        // (name, start_time, status) populate the runs list in the UI.
        let run_name = output_dir
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("run");
        let metadata_rows: [(&str, String); 4] = [
            ("active_session_id", format!("\"{}\"", session_id)),
            ("name", format!("\"{}\"", run_name.replace('"', "\\\""))),
            ("start_time", format!("{}", now)),
            ("status", "\"running\"".into()),
        ];
        for (k, v) in &metadata_rows {
            conn.execute(
                "INSERT OR REPLACE INTO metadata (key, value) VALUES (?, ?)",
                params![k, v],
            )?;
        }

        Ok(Self {
            conn: Mutex::new(conn),
            session_id,
            db_path,
        })
    }

    /// Generate a default session id from current time + random nibble.
    pub fn new_session_id() -> String {
        let now_ns = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        format!("run_{:x}", now_ns)
    }

    pub fn session_id(&self) -> &str {
        &self.session_id
    }

    /// Insert a scalar measurement. `step` is 1-indexed (post-step number).
    /// Uses `INSERT OR REPLACE` so re-running over the same step is idempotent.
    pub fn log_scalar(&self, tag: &str, step: u64, value: f64) {
        let wall = unix_now();
        if let Ok(conn) = self.conn.lock() {
            let _ = conn.execute(
                "INSERT OR REPLACE INTO scalars (tag, step, wall_time, value) VALUES (?, ?, ?, ?)",
                params![tag, step as i64, wall, value],
            );
        }
    }

    /// Batch-insert several scalars at the same step.
    pub fn log_scalars(&self, step: u64, kv: &[(&str, f64)]) {
        let wall = unix_now();
        if let Ok(conn) = self.conn.lock() {
            let tx = conn.unchecked_transaction();
            if let Ok(tx) = tx {
                {
                    let mut stmt = tx
                        .prepare_cached(
                            "INSERT OR REPLACE INTO scalars (tag, step, wall_time, value) VALUES (?, ?, ?, ?)",
                        )
                        .ok();
                    if let Some(stmt) = stmt.as_mut() {
                        for (tag, value) in kv {
                            let _ = stmt.execute(params![tag, step as i64, wall, value]);
                        }
                    }
                }
                let _ = tx.commit();
            }
        }
    }

    /// Update the session row's status. SerenityBoard schema constrains
    /// `status` to `running|complete|crashed`. We accept the colloquial
    /// "completed"/"failed" too and translate.
    pub fn set_status(&self, status: &str) {
        let canonical = match status {
            "completed" | "complete" => "complete",
            "failed" | "crashed" | "error" => "crashed",
            _ => "running",
        };
        if let Ok(conn) = self.conn.lock() {
            let _ = conn.execute(
                "UPDATE sessions SET status = ? WHERE session_id = ?",
                params![canonical, self.session_id],
            );
            let _ = conn.execute(
                "INSERT OR REPLACE INTO metadata (key, value) VALUES ('status', ?)",
                params![format!("\"{}\"", canonical)],
            );
        }
    }
}

impl Drop for BoardWriter {
    fn drop(&mut self) {
        // If status is still 'running' on drop, the trainer didn't call
        // set_status — treat as a crash. SerenityBoard's CHECK constraint
        // only allows running|complete|crashed.
        if let Ok(conn) = self.conn.lock() {
            let _ = conn.execute(
                "UPDATE sessions SET status = 'crashed' WHERE session_id = ? AND status = 'running'",
                params![self.session_id],
            );
            let _ = conn.execute(
                "INSERT OR REPLACE INTO metadata (key, value) VALUES ('status', '\"crashed\"')",
                [],
            );
        }
    }
}
