pub mod models;
mod responses;

use std::sync::Arc;

use axum::{
    Json, Router,
    extract::{Path, State},
    http::StatusCode,
    middleware,
    response::IntoResponse,
    routing::{get, post},
};
use tokio::sync::RwLock;

use crate::{VectorDatabase, errors::VectorDbError};

use self::models::{
    CollectionCreateRequest, CollectionRenameRequest, CollectionResponse, CollectionUpdateRequest,
    CollectionsResponse, EmbeddingRequest, EmbeddingResponse, ErrorResponse, SearchRequest,
    SearchResponse,
};
use self::responses::map_error;

#[derive(Clone)]
pub struct AppState {
    pub db: Arc<RwLock<VectorDatabase>>,
    pub auth_token: Option<String>,
}

pub type SharedState = AppState;

pub fn router(state: SharedState) -> Router {
    let protected_routes = Router::new()
        .route(
            "/collections",
            post(create_collection).get(list_collections),
        )
        .route(
            "/collections/:name",
            get(get_collection)
                .put(update_collection)
                .delete(delete_collection),
        )
        .route("/collections/:name/rename", post(rename_collection))
        .route("/collections/:name/vectors", post(insert_embedding))
        .route(
            "/collections/:name/vectors/:id",
            get(get_embedding)
                .put(update_embedding)
                .delete(delete_embedding),
        )
        .route("/collections/:name/search", post(search))
        .with_state(state.clone());

    let protected_routes = if state.auth_token.is_some() {
        protected_routes.layer(middleware::from_fn_with_state(state.clone(), require_auth))
    } else {
        protected_routes
    };

    Router::new()
        .route("/healthz", get(health))
        .merge(protected_routes)
        .with_state(state)
}

async fn health() -> &'static str {
    "ok"
}

async fn require_auth(
    State(state): State<SharedState>,
    request: axum::http::Request<axum::body::Body>,
    next: axum::middleware::Next,
) -> Result<impl IntoResponse, (StatusCode, Json<ErrorResponse>)> {
    if let Some(expected) = &state.auth_token {
        let unauthorized = || {
            (
                StatusCode::UNAUTHORIZED,
                Json(ErrorResponse::new(
                    "unauthorized",
                    "missing or invalid token",
                )),
            )
        };
        let expected_header = format!("Bearer {expected}");
        match request.headers().get(axum::http::header::AUTHORIZATION) {
            Some(value) => match value.to_str() {
                Ok(provided) if provided == expected_header => {}
                _ => return Err(unauthorized()),
            },
            None => return Err(unauthorized()),
        }
    }
    Ok(next.run(request).await)
}

async fn create_collection(
    State(state): State<SharedState>,
    Json(payload): Json<CollectionCreateRequest>,
) -> Result<impl IntoResponse, (StatusCode, Json<ErrorResponse>)> {
    let mut db = state.db.write().await;
    let CollectionCreateRequest {
        name,
        dimension,
        metric,
    } = payload;
    let metric = metric.unwrap_or(crate::distance::DistanceMetric::Euclidean);
    db.create_collection_with_metric(name.clone(), dimension, metric)
        .map_err(map_error)?;
    Ok((
        StatusCode::CREATED,
        Json(CollectionResponse::new(name, dimension, metric, 0)),
    ))
}

async fn list_collections(
    State(state): State<SharedState>,
) -> Result<impl IntoResponse, (StatusCode, Json<ErrorResponse>)> {
    let db = state.db.read().await;
    Ok((StatusCode::OK, Json(CollectionsResponse::from(&*db))))
}

async fn get_collection(
    Path(name): Path<String>,
    State(state): State<SharedState>,
) -> Result<impl IntoResponse, (StatusCode, Json<ErrorResponse>)> {
    let db = state.db.read().await;
    let collection = db
        .collection(&name)
        .ok_or_else(|| map_error(VectorDbError::CollectionNotFound(name.clone())))?;
    Ok((StatusCode::OK, Json(CollectionResponse::from(collection))))
}

async fn update_collection(
    Path(name): Path<String>,
    State(state): State<SharedState>,
    Json(payload): Json<CollectionUpdateRequest>,
) -> Result<impl IntoResponse, (StatusCode, Json<ErrorResponse>)> {
    let mut db = state.db.write().await;
    db.update_collection_dimension(&name, payload.dimension)
        .map_err(map_error)?;
    let collection = db
        .collection(&name)
        .ok_or_else(|| map_error(VectorDbError::CollectionNotFound(name.clone())))?;
    Ok((StatusCode::OK, Json(CollectionResponse::from(collection))))
}

async fn rename_collection(
    Path(name): Path<String>,
    State(state): State<SharedState>,
    Json(payload): Json<CollectionRenameRequest>,
) -> Result<impl IntoResponse, (StatusCode, Json<ErrorResponse>)> {
    let mut db = state.db.write().await;
    let CollectionRenameRequest { name: new_name } = payload;
    db.rename_collection(&name, new_name.clone())
        .map_err(map_error)?;
    let collection = db
        .collection(&new_name)
        .ok_or_else(|| map_error(VectorDbError::CollectionNotFound(new_name.clone())))?;
    Ok((StatusCode::OK, Json(CollectionResponse::from(collection))))
}

async fn delete_collection(
    Path(name): Path<String>,
    State(state): State<SharedState>,
) -> Result<impl IntoResponse, (StatusCode, Json<ErrorResponse>)> {
    let mut db = state.db.write().await;
    db.delete_collection(&name).map_err(map_error)?;
    Ok(StatusCode::NO_CONTENT)
}

async fn insert_embedding(
    Path(name): Path<String>,
    State(state): State<SharedState>,
    Json(payload): Json<EmbeddingRequest>,
) -> Result<impl IntoResponse, (StatusCode, Json<ErrorResponse>)> {
    let mut db = state.db.write().await;
    db.insert_embedding(&name, payload.to_embedding())
        .map_err(map_error)?;
    let embedding = db
        .embedding(&name, payload.id)
        .ok_or_else(|| map_error(VectorDbError::VectorNotFound(payload.id)))?;
    Ok((
        StatusCode::CREATED,
        Json(EmbeddingResponse::from(embedding)),
    ))
}

async fn update_embedding(
    Path((name, id)): Path<(String, u64)>,
    State(state): State<SharedState>,
    Json(payload): Json<EmbeddingRequest>,
) -> Result<impl IntoResponse, (StatusCode, Json<ErrorResponse>)> {
    if payload.id != id {
        return Err((
            StatusCode::BAD_REQUEST,
            Json(ErrorResponse::new(
                "invalid_request",
                "id in path and payload must match",
            )),
        ));
    }
    let mut db = state.db.write().await;
    db.update_embedding(&name, payload.to_embedding())
        .map_err(map_error)?;
    let embedding = db
        .embedding(&name, payload.id)
        .ok_or_else(|| map_error(VectorDbError::VectorNotFound(payload.id)))?;
    Ok((StatusCode::OK, Json(EmbeddingResponse::from(embedding))))
}

async fn get_embedding(
    Path((name, id)): Path<(String, u64)>,
    State(state): State<SharedState>,
) -> Result<impl IntoResponse, (StatusCode, Json<ErrorResponse>)> {
    let db = state.db.read().await;
    let embedding = db
        .embedding(&name, id)
        .ok_or_else(|| map_error(VectorDbError::VectorNotFound(id)))?;
    Ok((StatusCode::OK, Json(EmbeddingResponse::from(embedding))))
}

async fn delete_embedding(
    Path((name, id)): Path<(String, u64)>,
    State(state): State<SharedState>,
) -> Result<impl IntoResponse, (StatusCode, Json<ErrorResponse>)> {
    let mut db = state.db.write().await;
    db.delete(&name, id).map_err(map_error)?;
    Ok(StatusCode::NO_CONTENT)
}

async fn search(
    Path(name): Path<String>,
    State(state): State<SharedState>,
    Json(payload): Json<SearchRequest>,
) -> Result<impl IntoResponse, (StatusCode, Json<ErrorResponse>)> {
    let db = state.db.read().await;
    let results = db
        .search(&name, &payload.query, payload.k)
        .map_err(map_error)?;
    Ok((StatusCode::OK, Json(SearchResponse::from(results))))
}
