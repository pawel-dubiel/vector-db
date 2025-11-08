use axum::{Json, http::StatusCode};

use crate::errors::VectorDbError;

use super::models::ErrorResponse;

pub fn map_error(err: VectorDbError) -> (StatusCode, Json<ErrorResponse>) {
    let (status, code) = match &err {
        VectorDbError::CollectionAlreadyExists(_) => (StatusCode::CONFLICT, "collection_exists"),
        VectorDbError::CollectionNotFound(_) => (StatusCode::NOT_FOUND, "collection_not_found"),
        VectorDbError::InvalidCollectionName(_) => {
            (StatusCode::BAD_REQUEST, "invalid_collection_name")
        }
        VectorDbError::InvalidDimension => (StatusCode::BAD_REQUEST, "invalid_dimension"),
        VectorDbError::DimensionMismatch { .. } => (StatusCode::BAD_REQUEST, "dimension_mismatch"),
        VectorDbError::EmptyIndex => (StatusCode::BAD_REQUEST, "empty_index"),
        VectorDbError::InvalidSearchK { .. } => (StatusCode::BAD_REQUEST, "invalid_search_k"),
        VectorDbError::ZeroResultsRequested => (StatusCode::BAD_REQUEST, "zero_results"),
        VectorDbError::VectorContainsNaN => (StatusCode::BAD_REQUEST, "vector_contains_nan"),
        VectorDbError::VectorAlreadyExists(_) => (StatusCode::CONFLICT, "vector_exists"),
        VectorDbError::VectorNotFound(_) => (StatusCode::NOT_FOUND, "vector_not_found"),
        VectorDbError::CorruptedStorage(_) => {
            (StatusCode::INTERNAL_SERVER_ERROR, "corrupted_storage")
        }
        VectorDbError::InvalidStorageRoot(_) => {
            (StatusCode::INTERNAL_SERVER_ERROR, "invalid_storage_root")
        }
        VectorDbError::Io(_) => (StatusCode::INTERNAL_SERVER_ERROR, "io_error"),
    };

    (status, Json(ErrorResponse::new(code, err.to_string())))
}
