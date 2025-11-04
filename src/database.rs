use crate::collection::Collection;
use crate::embedding::{Embedding, VectorId};
use crate::errors::VectorDbError;
use crate::index::SearchResult;
use crate::storage::{
    COLLECTION_EXTENSION, collection_path, load_collection, persist_collection, replay_wal,
};
use crate::validation::validate_collection_name;
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};

#[derive(Debug)]
pub struct VectorDatabase {
    collections: HashMap<String, Collection>,
    storage_root: PathBuf,
}

impl VectorDatabase {
    pub fn open(path: impl AsRef<Path>) -> Result<Self, VectorDbError> {
        let path = path.as_ref();
        fs::create_dir_all(path)?;
        let metadata = fs::metadata(path)?;
        if !metadata.is_dir() {
            return Err(VectorDbError::InvalidStorageRoot(
                path.display().to_string(),
            ));
        }

        replay_wal(path)?;

        let mut database = Self {
            collections: HashMap::new(),
            storage_root: path.to_path_buf(),
        };
        database.load_collections()?;
        Ok(database)
    }

    pub fn create_collection(
        &mut self,
        name: impl Into<String>,
        dimension: usize,
    ) -> Result<(), VectorDbError> {
        let name = name.into();
        validate_collection_name(&name)?;
        if self.collections.contains_key(&name) {
            return Err(VectorDbError::CollectionAlreadyExists(name));
        }
        let collection = Collection::new(name.clone(), dimension)?;
        persist_collection(&collection, &self.storage_root)?;
        self.collections.insert(name, collection);
        Ok(())
    }

    pub fn insert(
        &mut self,
        collection: &str,
        id: VectorId,
        vector: Vec<f32>,
    ) -> Result<(), VectorDbError> {
        self.apply_and_persist(collection, move |col| col.insert(id, vector))
    }

    pub fn insert_embedding(
        &mut self,
        collection: &str,
        embedding: Embedding,
    ) -> Result<(), VectorDbError> {
        self.apply_and_persist(collection, move |col| col.insert_embedding(embedding))
    }

    pub fn update_embedding(
        &mut self,
        collection: &str,
        embedding: Embedding,
    ) -> Result<(), VectorDbError> {
        self.apply_and_persist(collection, move |col| col.update_embedding(embedding))
    }

    pub fn update_vector(
        &mut self,
        collection: &str,
        id: VectorId,
        vector: Vec<f32>,
    ) -> Result<(), VectorDbError> {
        self.apply_and_persist(collection, move |col| col.update_vector(id, vector))
    }

    pub fn delete(&mut self, collection: &str, id: VectorId) -> Result<(), VectorDbError> {
        self.apply_and_persist(collection, move |col| col.delete(id))
    }

    pub fn search(
        &self,
        collection: &str,
        query: &[f32],
        k: usize,
    ) -> Result<Vec<SearchResult<'_>>, VectorDbError> {
        let collection_ref = self
            .collections
            .get(collection)
            .ok_or_else(|| VectorDbError::CollectionNotFound(collection.to_string()))?;
        collection_ref.search(query, k)
    }

    pub fn collection(&self, name: &str) -> Option<&Collection> {
        self.collections.get(name)
    }

    pub fn embedding(&self, collection: &str, id: VectorId) -> Option<&Embedding> {
        self.collections.get(collection)?.embedding(id)
    }

    fn apply_and_persist<F>(&mut self, name: &str, action: F) -> Result<(), VectorDbError>
    where
        F: FnOnce(&mut Collection) -> Result<(), VectorDbError>,
    {
        let persist_result = {
            let collection = self
                .collections
                .get_mut(name)
                .ok_or_else(|| VectorDbError::CollectionNotFound(name.to_string()))?;
            action(collection)?;
            persist_collection(collection, &self.storage_root)
        };

        if let Err(persist_err) = persist_result {
            let path = collection_path(&self.storage_root, name);
            let reload_result = if path.exists() {
                load_collection(name, &path)
            } else {
                Err(VectorDbError::CorruptedStorage(format!(
                    "collection file '{}' missing during recovery",
                    path.display()
                )))
            };

            match reload_result {
                Ok(reloaded) => {
                    if let Some(slot) = self.collections.get_mut(name) {
                        *slot = reloaded;
                    }
                    Err(persist_err)
                }
                Err(reload_err) => {
                    self.collections.remove(name);
                    Err(reload_err)
                }
            }
        } else {
            Ok(())
        }
    }

    fn load_collections(&mut self) -> Result<(), VectorDbError> {
        for entry in fs::read_dir(&self.storage_root)? {
            let entry = entry?;
            let path = entry.path();
            if !path.is_file() {
                continue;
            }
            if path.extension().and_then(|ext| ext.to_str()) != Some(COLLECTION_EXTENSION) {
                continue;
            }
            let name = path
                .file_stem()
                .and_then(|stem| stem.to_str())
                .ok_or_else(|| {
                    VectorDbError::CorruptedStorage("invalid collection filename".into())
                })?
                .to_string();
            let collection = load_collection(&name, &path)?;
            self.collections.insert(name, collection);
        }
        Ok(())
    }
}
