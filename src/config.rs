use clap::{Parser, ValueEnum};
use std::fmt;
use std::net::SocketAddr;
use std::path::PathBuf;

const DEFAULT_BIND: &str = "127.0.0.1:8080";

/// Configuration for the VectorDB HTTP server.
#[derive(Debug, Clone, Parser)]
#[command(author, version, about = "HTTP server exposing the VectorDB API")]
pub struct ServerConfig {
    /// Address to bind the HTTP server to.
    #[arg(long, env = "VECTORDB_BIND", default_value = DEFAULT_BIND)]
    pub bind: SocketAddr,

    /// Path to the directory used for on-disk storage.
    #[arg(
        long,
        env = "VECTORDB_STORAGE",
        value_name = "PATH",
        default_value = "./vectordb-data"
    )]
    pub storage: PathBuf,

    /// Optional static bearer token that clients must provide.
    #[arg(long, env = "VECTORDB_AUTH_TOKEN")]
    pub auth_token: Option<String>,

    /// Minimum log level for the server.
    #[arg(long, env = "VECTORDB_LOG", default_value_t = LogLevel::Info)]
    pub log_level: LogLevel,
}

#[derive(Debug, Clone, Copy, ValueEnum)]
pub enum LogLevel {
    Error,
    Warn,
    Info,
    Debug,
    Trace,
}

impl LogLevel {
    pub fn into_filter(self) -> &'static str {
        match self {
            LogLevel::Error => "error",
            LogLevel::Warn => "warn",
            LogLevel::Info => "info",
            LogLevel::Debug => "debug",
            LogLevel::Trace => "trace",
        }
    }
}

impl fmt::Display for LogLevel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.into_filter())
    }
}
