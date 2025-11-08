use std::sync::Arc;

use clap::Parser;
use tokio::{signal, sync::RwLock};
use tracing::{error, info};
use tracing_subscriber::{EnvFilter, fmt};
use vectordb::{
    VectorDatabase,
    api::{self, AppState},
    config::ServerConfig,
};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let config = ServerConfig::parse();

    let env_filter = EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| EnvFilter::new(config.log_level.into_filter()));

    fmt().with_env_filter(env_filter).init();

    info!("opening database", path = ?config.storage);
    let db = match VectorDatabase::open(&config.storage) {
        Ok(db) => db,
        Err(err) => {
            error!(error = %err, "failed to open database");
            return Err(err.into());
        }
    };

    let state = AppState {
        db: Arc::new(RwLock::new(db)),
        auth_token: config.auth_token.clone(),
    };

    let app = api::router(state);

    let listener = tokio::net::TcpListener::bind(config.bind).await?;
    info!(address = %config.bind, "server listening");

    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal())
        .await?;

    info!("server shutdown complete");
    Ok(())
}

async fn shutdown_signal() {
    let ctrl_c = async {
        signal::ctrl_c()
            .await
            .expect("failed to install Ctrl+C handler");
    };

    #[cfg(unix)]
    let terminate = async {
        signal::unix::signal(signal::unix::SignalKind::terminate())
            .expect("failed to install signal handler")
            .recv()
            .await;
    };

    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();

    tokio::select! {
        _ = ctrl_c => {},
        _ = terminate => {},
    }
}
