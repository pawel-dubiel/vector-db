use std::error::Error as StdError;
use std::fmt;

use crate::embedding::VectorId;

#[derive(Debug)]
pub enum VectorDbError {
    CollectionAlreadyExists(String),
    CollectionNotFound(String),
    CollectionNotEmpty(String),
    InvalidCollectionName(String),
    InvalidDimension,
    DimensionMismatch { expected: usize, found: usize },
    EmptyIndex,
    InvalidSearchK { requested: usize, available: usize },
    ZeroResultsRequested,
    VectorContainsNaN,
    VectorAlreadyExists(VectorId),
    VectorNotFound(VectorId),
    CorruptedStorage(String),
    InvalidStorageRoot(String),
    Io(std::io::Error),
}

impl fmt::Display for VectorDbError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            VectorDbError::CollectionAlreadyExists(name) => {
                write!(f, "collection '{name}' already exists")
            }
            VectorDbError::CollectionNotFound(name) => {
                write!(f, "collection '{name}' not found")
            }
            VectorDbError::CollectionNotEmpty(name) => {
                write!(f, "collection '{name}' is not empty")
            }
            VectorDbError::InvalidCollectionName(name) => {
                write!(f, "collection name '{name}' is not allowed")
            }
            VectorDbError::InvalidDimension => write!(f, "dimension must be greater than zero"),
            VectorDbError::DimensionMismatch { expected, found } => write!(
                f,
                "vector dimension mismatch: expected {expected}, found {found}"
            ),
            VectorDbError::EmptyIndex => write!(f, "cannot search empty index"),
            VectorDbError::InvalidSearchK {
                requested,
                available,
            } => write!(
                f,
                "requested {requested} results but only {available} vectors indexed"
            ),
            VectorDbError::ZeroResultsRequested => write!(f, "search requires at least one result"),
            VectorDbError::VectorContainsNaN => write!(f, "vector contains NaN values"),
            VectorDbError::VectorAlreadyExists(id) => {
                write!(f, "vector with id {id} already exists in the index")
            }
            VectorDbError::VectorNotFound(id) => {
                write!(f, "vector with id {id} not found in the index")
            }
            VectorDbError::CorruptedStorage(msg) => write!(f, "corrupted storage: {msg}"),
            VectorDbError::InvalidStorageRoot(path) => {
                write!(f, "storage root '{path}' is not a directory")
            }
            VectorDbError::Io(err) => write!(f, "io error: {err}"),
        }
    }
}

impl StdError for VectorDbError {
    fn source(&self) -> Option<&(dyn StdError + 'static)> {
        match self {
            VectorDbError::Io(err) => Some(err),
            _ => None,
        }
    }
}

impl From<std::io::Error> for VectorDbError {
    fn from(error: std::io::Error) -> Self {
        VectorDbError::Io(error)
    }
}
