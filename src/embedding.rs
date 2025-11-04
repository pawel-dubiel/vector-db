use std::collections::BTreeMap;

pub type VectorId = u64;
pub type Metadata = BTreeMap<String, String>;

#[derive(Debug, Clone, PartialEq)]
pub struct Embedding {
    pub id: VectorId,
    pub vector: Vec<f32>,
    pub metadata: Metadata,
}

impl Embedding {
    pub fn new(id: VectorId, vector: Vec<f32>, metadata: Metadata) -> Self {
        Self {
            id,
            vector,
            metadata,
        }
    }
}
