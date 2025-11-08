//! A minimal on-disk vector database with a flat index.
//!
//! The database supports multiple named collections. Each collection enforces a
//! fixed dimensionality for the vectors it stores and exposes brute-force
//! similarity search via a flat index. All collections are durably stored on
//! disk so that reopening the database reloads the full state.
//!
//! # Examples
//! ```
//! use std::collections::BTreeMap;
//! use std::fs;
//! use std::time::{SystemTime, UNIX_EPOCH};
//! use vectordb::{Embedding, VectorDatabase, VectorDbError};
//!
//! # fn main() -> Result<(), VectorDbError> {
//! let dir = std::env::temp_dir().join(format!(
//!     "vectordb_example_{}",
//!     SystemTime::now()
//!         .duration_since(UNIX_EPOCH)
//!         .expect("clock drift")
//!         .as_nanos()
//! ));
//!
//! let mut db = VectorDatabase::open(&dir)?;
//! db.create_collection("documents", 3)?;
//!
//! let mut metadata = BTreeMap::new();
//! metadata.insert("title".into(), "Doc A".into());
//! db.insert_embedding("documents", Embedding::new(7, vec![0.0, 1.0, 0.5], metadata))?;
//!
//! let results = db.search("documents", &[0.0, 1.0, 0.4], 1)?;
//! assert_eq!(results[0].embedding.metadata.get("title"), Some(&"Doc A".to_string()));
//!
//! drop(db);
//! fs::remove_dir_all(dir)?;
//! # Ok(())
//! # }
//! ```

mod collection;
mod database;
mod distance;
mod embedding;
mod errors;
mod index;
mod storage;
mod validation;

pub mod api;
pub mod config;

pub use crate::database::VectorDatabase;
pub use crate::embedding::{Embedding, Metadata, VectorId};
pub use crate::errors::VectorDbError;
pub use crate::index::SearchResult;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::collection::Collection;
    use crate::storage::{COLLECTION_EXTENSION, collection_path, encode_collection};
    use std::collections::BTreeMap;
    use std::fs;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn temp_dir(name: &str) -> std::path::PathBuf {
        std::env::temp_dir().join(format!(
            "vectordb_test_{}_{}",
            name,
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .expect("clock drift")
                .as_nanos()
        ))
    }

    fn cleanup_dir(path: std::path::PathBuf) {
        if path.exists() {
            fs::remove_dir_all(path).unwrap();
        }
    }

    #[test]
    fn creating_collections_with_invalid_dimension_fails() {
        let dir = temp_dir("invalid_dimension");
        let mut db = VectorDatabase::open(&dir).unwrap();
        let err = db.create_collection("users", 0).unwrap_err();
        assert!(matches!(err, VectorDbError::InvalidDimension));
        drop(db);
        cleanup_dir(dir);
    }

    #[test]
    fn adding_duplicate_collection_name_fails() {
        let dir = temp_dir("duplicate_collection");
        let mut db = VectorDatabase::open(&dir).unwrap();
        db.create_collection("items", 3).unwrap();
        let err = db.create_collection("items", 3).unwrap_err();
        match err {
            VectorDbError::CollectionAlreadyExists(name) => assert_eq!(name, "items"),
            other => panic!("unexpected error: {other}"),
        }
        drop(db);
        cleanup_dir(dir);
    }

    #[test]
    fn insert_and_search_round_trip() {
        let dir = temp_dir("round_trip");
        let mut db = VectorDatabase::open(&dir).unwrap();
        db.create_collection("embeddings", 2).unwrap();
        db.insert("embeddings", 1, vec![1.0, 0.0]).unwrap();
        db.insert("embeddings", 2, vec![0.0, 1.0]).unwrap();
        db.insert("embeddings", 3, vec![1.0, 1.0]).unwrap();

        let results = db.search("embeddings", &[1.0, 0.0], 2).unwrap();
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].id, 1);
        assert!(results[0].distance <= results[1].distance);
        drop(db);
        cleanup_dir(dir);
    }

    #[test]
    fn search_with_empty_index_fails() {
        let dir = temp_dir("empty_search");
        let mut db = VectorDatabase::open(&dir).unwrap();
        db.create_collection("vectors", 2).unwrap();
        let err = db.search("vectors", &[0.0, 0.0], 1).unwrap_err();
        assert!(matches!(err, VectorDbError::EmptyIndex));
        drop(db);
        cleanup_dir(dir);
    }

    #[test]
    fn insert_with_dimension_mismatch_fails() {
        let dir = temp_dir("dimension_mismatch");
        let mut db = VectorDatabase::open(&dir).unwrap();
        db.create_collection("vectors", 3).unwrap();
        let err = db.insert("vectors", 1, vec![1.0, 2.0]).unwrap_err();
        assert!(matches!(
            err,
            VectorDbError::DimensionMismatch {
                expected: 3,
                found: 2
            }
        ));
        drop(db);
        cleanup_dir(dir);
    }

    #[test]
    fn search_with_k_greater_than_entries_fails() {
        let dir = temp_dir("invalid_k");
        let mut db = VectorDatabase::open(&dir).unwrap();
        db.create_collection("vectors", 2).unwrap();
        db.insert("vectors", 1, vec![1.0, 1.0]).unwrap();
        let err = db.search("vectors", &[0.0, 0.0], 2).unwrap_err();
        assert!(matches!(
            err,
            VectorDbError::InvalidSearchK {
                requested: 2,
                available: 1
            }
        ));
        drop(db);
        cleanup_dir(dir);
    }

    #[test]
    fn search_with_nan_vector_fails() {
        let dir = temp_dir("nan_query");
        let mut db = VectorDatabase::open(&dir).unwrap();
        db.create_collection("vectors", 2).unwrap();
        db.insert("vectors", 1, vec![1.0, 1.0]).unwrap();
        let err = db.search("vectors", &[f32::NAN, 0.0], 1).unwrap_err();
        assert!(matches!(err, VectorDbError::VectorContainsNaN));
        drop(db);
        cleanup_dir(dir);
    }

    #[test]
    fn embedding_metadata_round_trip() {
        let dir = temp_dir("metadata_round_trip");
        let mut db = VectorDatabase::open(&dir).unwrap();
        db.create_collection("docs", 3).unwrap();

        let mut metadata = BTreeMap::new();
        metadata.insert("title".into(), "Doc".into());

        db.insert_embedding("docs", Embedding::new(1, vec![0.1, 0.2, 0.3], metadata))
            .unwrap();

        let results = db.search("docs", &[0.1, 0.2, 0.3], 1).unwrap();
        assert_eq!(results[0].embedding.metadata.get("title").unwrap(), "Doc");
        drop(db);
        cleanup_dir(dir);
    }

    #[test]
    fn duplicate_vector_id_fails() {
        let dir = temp_dir("duplicate_vector");
        let mut db = VectorDatabase::open(&dir).unwrap();
        db.create_collection("dup", 2).unwrap();
        db.insert("dup", 1, vec![0.0, 0.0]).unwrap();
        let err = db.insert("dup", 1, vec![0.1, 0.2]).unwrap_err();
        assert!(matches!(err, VectorDbError::VectorAlreadyExists(1)));
        drop(db);
        cleanup_dir(dir);
    }

    #[test]
    fn reopening_preserves_embeddings() {
        let dir = temp_dir("persist");
        {
            let mut db = VectorDatabase::open(&dir).unwrap();
            db.create_collection("docs", 2).unwrap();
            let mut metadata = BTreeMap::new();
            metadata.insert("title".into(), "Persisted".into());
            db.insert_embedding("docs", Embedding::new(5, vec![0.4, 0.5], metadata))
                .unwrap();
        }

        let db = VectorDatabase::open(&dir).unwrap();
        let embedding = db.embedding("docs", 5).expect("embedding should persist");
        assert_eq!(embedding.vector, vec![0.4, 0.5]);
        assert_eq!(embedding.metadata.get("title").unwrap(), "Persisted");
        drop(db);
        cleanup_dir(dir);
    }

    #[test]
    fn pending_wal_is_replayed_on_open() {
        let dir = temp_dir("wal_replay");
        {
            let mut db = VectorDatabase::open(&dir).unwrap();
            db.create_collection("docs", 2).unwrap();
            db.insert("docs", 1, vec![0.0, 0.0]).unwrap();
        }

        let mut collection = Collection::new("docs".into(), 2).unwrap();
        collection.insert(1, vec![0.0, 0.0]).unwrap();
        collection.insert(2, vec![0.5, 0.5]).unwrap();

        let wal_bytes = encode_collection(&collection).unwrap();
        let wal_path =
            collection_path(&dir, "docs").with_extension(format!("{COLLECTION_EXTENSION}.wal"));
        fs::write(&wal_path, wal_bytes).unwrap();

        let db = VectorDatabase::open(&dir).unwrap();
        assert!(db.embedding("docs", 2).is_some());
        assert!(!wal_path.exists());
        drop(db);
        cleanup_dir(dir);
    }

    #[test]
    fn updating_embedding_replaces_vector_and_metadata() {
        let dir = temp_dir("update_embedding");
        let mut db = VectorDatabase::open(&dir).unwrap();
        db.create_collection("docs", 2).unwrap();

        db.insert("docs", 1, vec![0.0, 0.0]).unwrap();

        let mut metadata = BTreeMap::new();
        metadata.insert("title".into(), "Updated".into());
        db.update_embedding("docs", Embedding::new(1, vec![1.0, 1.0], metadata))
            .unwrap();

        let embedding = db.embedding("docs", 1).unwrap();
        assert_eq!(embedding.vector, vec![1.0, 1.0]);
        assert_eq!(embedding.metadata.get("title").unwrap(), "Updated");
        drop(db);
        cleanup_dir(dir);
    }

    #[test]
    fn deleting_embedding_removes_it_from_disk() {
        let dir = temp_dir("delete_embedding");
        {
            let mut db = VectorDatabase::open(&dir).unwrap();
            db.create_collection("docs", 2).unwrap();
            db.insert("docs", 1, vec![0.0, 0.0]).unwrap();
            db.insert("docs", 2, vec![1.0, 1.0]).unwrap();
            db.delete("docs", 1).unwrap();
        }

        let db = VectorDatabase::open(&dir).unwrap();
        assert!(db.embedding("docs", 1).is_none());
        assert!(db.embedding("docs", 2).is_some());
        drop(db);
        cleanup_dir(dir);
    }

    #[test]
    fn updating_or_deleting_nonexistent_vector_fails() {
        let dir = temp_dir("update_delete_fail");
        let mut db = VectorDatabase::open(&dir).unwrap();
        db.create_collection("docs", 2).unwrap();
        let err = db.delete("docs", 42).unwrap_err();
        assert!(matches!(err, VectorDbError::VectorNotFound(42)));

        let mut metadata = BTreeMap::new();
        metadata.insert("title".into(), "Missing".into());
        let err = db
            .update_embedding("docs", Embedding::new(99, vec![0.0, 0.0], metadata))
            .unwrap_err();
        assert!(matches!(err, VectorDbError::VectorNotFound(99)));
        drop(db);
        cleanup_dir(dir);
    }
}
