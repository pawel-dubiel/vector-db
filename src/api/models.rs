use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use crate::{VectorDatabase, collection::Collection, embedding::Embedding, index::SearchResult};

#[derive(Debug, Deserialize)]
pub struct CollectionCreateRequest {
    pub name: String,
    pub dimension: usize,
}

#[derive(Debug, Deserialize)]
pub struct CollectionRenameRequest {
    pub name: String,
}

#[derive(Debug, Deserialize)]
pub struct CollectionUpdateRequest {
    pub dimension: usize,
}

#[derive(Debug, Serialize)]
pub struct CollectionResponse {
    pub name: String,
    pub dimension: usize,
    pub vectors: usize,
}

impl CollectionResponse {
    pub fn new(name: impl Into<String>, dimension: usize, vectors: usize) -> Self {
        Self {
            name: name.into(),
            dimension,
            vectors,
        }
    }
}

impl From<&Collection> for CollectionResponse {
    fn from(collection: &Collection) -> Self {
        Self::new(
            collection.name(),
            collection.dimension(),
            collection.embeddings().len(),
        )
    }
}

impl From<(&str, &Collection)> for CollectionResponse {
    fn from((name, collection): (&str, &Collection)) -> Self {
        Self::new(name, collection.dimension(), collection.embeddings().len())
    }
}

#[derive(Debug, Serialize)]
pub struct CollectionsResponse {
    pub collections: Vec<CollectionResponse>,
}

impl From<&VectorDatabase> for CollectionsResponse {
    fn from(db: &VectorDatabase) -> Self {
        let collections = db
            .collections()
            .map(|(name, collection)| CollectionResponse::from((name.as_str(), collection)))
            .collect();
        Self { collections }
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct EmbeddingRequest {
    pub id: u64,
    pub vector: Vec<f32>,
    #[serde(default)]
    pub metadata: Option<BTreeMap<String, String>>,
}

impl EmbeddingRequest {
    pub fn to_embedding(&self) -> Embedding {
        Embedding::new(
            self.id,
            self.vector.clone(),
            self.metadata.clone().unwrap_or_default(),
        )
    }
}

#[derive(Debug, Serialize)]
pub struct EmbeddingResponse {
    pub id: u64,
    pub vector: Vec<f32>,
    pub metadata: BTreeMap<String, String>,
}

impl From<&Embedding> for EmbeddingResponse {
    fn from(embedding: &Embedding) -> Self {
        Self {
            id: embedding.id,
            vector: embedding.vector.clone(),
            metadata: embedding.metadata.clone(),
        }
    }
}

#[derive(Debug, Deserialize)]
pub struct SearchRequest {
    pub query: Vec<f32>,
    pub k: usize,
}

#[derive(Debug, Serialize)]
pub struct SearchResponse {
    pub results: Vec<SearchResultResponse>,
}

impl From<Vec<SearchResult<'_>>> for SearchResponse {
    fn from(results: Vec<SearchResult<'_>>) -> Self {
        Self {
            results: results
                .into_iter()
                .map(SearchResultResponse::from)
                .collect(),
        }
    }
}

#[derive(Debug, Serialize)]
pub struct SearchResultResponse {
    pub id: u64,
    pub distance: f32,
    pub embedding: EmbeddingResponse,
}

impl From<SearchResult<'_>> for SearchResultResponse {
    fn from(result: SearchResult<'_>) -> Self {
        Self {
            id: result.id,
            distance: result.distance,
            embedding: EmbeddingResponse::from(result.embedding),
        }
    }
}

#[derive(Debug, Serialize)]
pub struct ErrorResponse {
    pub code: String,
    pub message: String,
}

impl ErrorResponse {
    pub fn new(code: impl Into<String>, message: impl Into<String>) -> Self {
        Self {
            code: code.into(),
            message: message.into(),
        }
    }
}
