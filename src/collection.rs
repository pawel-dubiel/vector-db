use crate::embedding::{Embedding, Metadata, VectorId};
use crate::errors::VectorDbError;
use crate::index::{FlatIndex, SearchResult};

#[derive(Debug, Clone)]
pub struct Collection {
    name: String,
    index: FlatIndex,
}

impl Collection {
    pub(crate) fn new(name: String, dimension: usize) -> Result<Self, VectorDbError> {
        Ok(Self {
            name,
            index: FlatIndex::new(dimension)?,
        })
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn dimension(&self) -> usize {
        self.index.dimension()
    }

    pub fn insert(&mut self, id: VectorId, vector: Vec<f32>) -> Result<(), VectorDbError> {
        self.index.add(id, vector, Metadata::new())
    }

    pub fn insert_embedding(&mut self, embedding: Embedding) -> Result<(), VectorDbError> {
        self.index.add_embedding(embedding)
    }

    pub fn update_embedding(&mut self, embedding: Embedding) -> Result<(), VectorDbError> {
        self.index.update_embedding(embedding)
    }

    pub fn update_vector(&mut self, id: VectorId, vector: Vec<f32>) -> Result<(), VectorDbError> {
        self.index.update_vector(id, vector)
    }

    pub fn delete(&mut self, id: VectorId) -> Result<(), VectorDbError> {
        self.index.remove(id)
    }

    pub fn search(&self, query: &[f32], k: usize) -> Result<Vec<SearchResult<'_>>, VectorDbError> {
        self.index.search(query, k)
    }

    pub fn embedding(&self, id: VectorId) -> Option<&Embedding> {
        self.index.get(id)
    }

    pub(crate) fn embeddings(&self) -> &[Embedding] {
        self.index.embeddings()
    }
}
