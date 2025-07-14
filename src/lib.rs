pub mod trainers;
pub mod loaders;
pub mod models;
pub mod memory;

// Re-export common types
pub use trainers::{Config, ProcessConfig, load_config};

pub mod logging {
    use log::LevelFilter;
    use env_logger::Builder;
    use std::io::Write;
    
    pub fn init_logger() {
        Builder::new()
            .format(|buf, record| {
                writeln!(
                    buf,
                    "{} [{}] - {}",
                    chrono::Local::now().format("%Y-%m-%d %H:%M:%S"),
                    record.level(),
                    record.args()
                )
            })
            .filter(None, LevelFilter::Info)
            .init();
    }
}