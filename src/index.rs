use crate::distance::DistanceMetric;
use crate::embedding::{Embedding, Metadata, VectorId};
use crate::errors::VectorDbError;

#[derive(Debug)]
pub struct SearchResult<'a> {
    pub id: VectorId,
    pub distance: f32,
    pub embedding: &'a Embedding,
}

#[derive(Debug, Clone)]
pub(crate) struct FlatIndex {
    dimension: usize,
    metric: DistanceMetric,
    entries: Vec<Embedding>,
}

impl FlatIndex {
    pub(crate) fn new(dimension: usize, metric: DistanceMetric) -> Result<Self, VectorDbError> {
        if dimension == 0 {
            return Err(VectorDbError::InvalidDimension);
        }
        Ok(Self {
            dimension,
            metric,
            entries: Vec::new(),
        })
    }

    pub(crate) fn add(
        &mut self,
        id: VectorId,
        vector: Vec<f32>,
        metadata: Metadata,
    ) -> Result<(), VectorDbError> {
        let embedding = Embedding::new(id, vector, metadata);
        self.add_embedding(embedding)
    }

    pub(crate) fn add_embedding(&mut self, embedding: Embedding) -> Result<(), VectorDbError> {
        self.validate_vector(&embedding.vector)?;
        if self.entries.iter().any(|entry| entry.id == embedding.id) {
            return Err(VectorDbError::VectorAlreadyExists(embedding.id));
        }
        self.entries.push(embedding);
        Ok(())
    }

    pub(crate) fn update_embedding(&mut self, embedding: Embedding) -> Result<(), VectorDbError> {
        self.validate_vector(&embedding.vector)?;
        match self
            .entries
            .iter_mut()
            .find(|entry| entry.id == embedding.id)
        {
            Some(existing) => {
                *existing = embedding;
                Ok(())
            }
            None => Err(VectorDbError::VectorNotFound(embedding.id)),
        }
    }

    pub(crate) fn update_vector(
        &mut self,
        id: VectorId,
        vector: Vec<f32>,
    ) -> Result<(), VectorDbError> {
        self.validate_vector(&vector)?;
        match self.entries.iter_mut().find(|entry| entry.id == id) {
            Some(existing) => {
                existing.vector = vector;
                Ok(())
            }
            None => Err(VectorDbError::VectorNotFound(id)),
        }
    }

    pub(crate) fn remove(&mut self, id: VectorId) -> Result<(), VectorDbError> {
        if let Some(pos) = self.entries.iter().position(|entry| entry.id == id) {
            self.entries.remove(pos);
            Ok(())
        } else {
            Err(VectorDbError::VectorNotFound(id))
        }
    }

    pub(crate) fn search(
        &self,
        query: &[f32],
        k: usize,
    ) -> Result<Vec<SearchResult<'_>>, VectorDbError> {
        if self.entries.is_empty() {
            return Err(VectorDbError::EmptyIndex);
        }
        if k == 0 {
            return Err(VectorDbError::ZeroResultsRequested);
        }
        if k > self.entries.len() {
            return Err(VectorDbError::InvalidSearchK {
                requested: k,
                available: self.entries.len(),
            });
        }
        self.validate_vector(query)?;

        let mut scores: Vec<SearchResult<'_>> = self
            .entries
            .iter()
            .map(|embedding| SearchResult {
                id: embedding.id,
                distance: self.metric.score(query, &embedding.vector),
                embedding,
            })
            .collect();

        scores.sort_by(|a, b| {
            a.distance
                .partial_cmp(&b.distance)
                .expect("distance must not be NaN")
        });
        scores.truncate(k);
        Ok(scores)
    }

    pub(crate) fn get(&self, id: VectorId) -> Option<&Embedding> {
        self.entries.iter().find(|entry| entry.id == id)
    }

    pub(crate) fn dimension(&self) -> usize {
        self.dimension
    }

    pub(crate) fn metric(&self) -> DistanceMetric {
        self.metric
    }

    pub(crate) fn embeddings(&self) -> &[Embedding] {
        &self.entries
    }

    fn validate_vector(&self, vector: &[f32]) -> Result<(), VectorDbError> {
        if vector.len() != self.dimension {
            return Err(VectorDbError::DimensionMismatch {
                expected: self.dimension,
                found: vector.len(),
            });
        }
        if vector.iter().any(|v| v.is_nan()) {
            return Err(VectorDbError::VectorContainsNaN);
        }
        Ok(())
    }
}
