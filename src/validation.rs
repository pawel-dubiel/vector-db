use crate::errors::VectorDbError;

pub(crate) fn validate_collection_name(name: &str) -> Result<(), VectorDbError> {
    if name.is_empty() || name.contains('/') || name.contains('\\') || name.contains('\0') {
        return Err(VectorDbError::InvalidCollectionName(name.to_string()));
    }
    Ok(())
}
