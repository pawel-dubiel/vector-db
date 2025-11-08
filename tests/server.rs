use std::sync::Arc;

use axum::{
    Router,
    body::Body,
    http::{Request, StatusCode},
};
use serde_json::{Value, json};
use tempfile::TempDir;
use tokio::sync::RwLock;
use tower::ServiceExt;
use vectordb::{
    VectorDatabase,
    api::{self, AppState},
};

fn test_app(auth_token: Option<&str>) -> (Router, TempDir) {
    let temp_dir = TempDir::new().expect("temp dir");
    let db = VectorDatabase::open(temp_dir.path()).expect("open db");
    let state = AppState {
        db: Arc::new(RwLock::new(db)),
        auth_token: auth_token.map(|token| token.to_string()),
    };
    (api::router(state), temp_dir)
}

async fn read_body(response: axum::response::Response) -> Value {
    let bytes = hyper::body::to_bytes(response.into_body())
        .await
        .expect("body bytes");
    serde_json::from_slice(&bytes).expect("json response")
}

#[tokio::test]
async fn create_collection_and_insert_embedding() {
    let (app, _dir) = test_app(None);

    let response = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/collections")
                .header("content-type", "application/json")
                .body(Body::from(
                    json!({
                        "name": "documents",
                        "dimension": 3
                    })
                    .to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::CREATED);

    let response = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/collections/documents/vectors")
                .header("content-type", "application/json")
                .body(Body::from(
                    json!({
                        "id": 1,
                        "vector": [0.0, 1.0, 0.5],
                        "metadata": {"title": "Doc A"}
                    })
                    .to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::CREATED);
    let body = read_body(response).await;
    assert_eq!(body["id"], 1);
    assert_eq!(body["metadata"]["title"], "Doc A");

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/collections/documents/search")
                .header("content-type", "application/json")
                .body(Body::from(
                    json!({
                        "query": [0.0, 1.0, 0.4],
                        "k": 1
                    })
                    .to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    let body = read_body(response).await;
    assert_eq!(body["results"][0]["id"], 1);
}

#[tokio::test]
async fn creating_collection_with_invalid_dimension_returns_error() {
    let (app, _dir) = test_app(None);

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/collections")
                .header("content-type", "application/json")
                .body(Body::from(
                    json!({
                        "name": "bad",
                        "dimension": 0
                    })
                    .to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    let body = read_body(response).await;
    assert_eq!(body["code"], "invalid_dimension");
}

#[tokio::test]
async fn authentication_is_enforced_when_configured() {
    let (app, _dir) = test_app(Some("secret"));

    let response = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/collections")
                .header("content-type", "application/json")
                .body(Body::from(
                    json!({
                        "name": "secure",
                        "dimension": 2
                    })
                    .to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::UNAUTHORIZED);

    let authed = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/collections")
                .header("content-type", "application/json")
                .header("authorization", "Bearer secret")
                .body(Body::from(
                    json!({
                        "name": "secure",
                        "dimension": 2
                    })
                    .to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(authed.status(), StatusCode::CREATED);
}
