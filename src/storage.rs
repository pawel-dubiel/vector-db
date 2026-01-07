use crate::collection::Collection;
use crate::embedding::{Embedding, Metadata};
use crate::errors::VectorDbError;
use crate::validation::validate_collection_name;
use std::fs;
use std::fs::File;
use std::io::{Cursor, Read, Write};
use std::path::{Path, PathBuf};

pub(crate) const COLLECTION_EXTENSION: &str = "vdb";
const WAL_SUFFIX: &str = "wal";
const FILE_MAGIC: [u8; 4] = *b"VDB1";
const FILE_VERSION: u16 = 1;

pub(crate) fn collection_path(root: &Path, name: &str) -> PathBuf {
    let mut path = root.to_path_buf();
    path.push(format!("{name}.{COLLECTION_EXTENSION}"));
    path
}

fn wal_path(root: &Path, name: &str) -> PathBuf {
    let extension = format!("{COLLECTION_EXTENSION}.{WAL_SUFFIX}");
    collection_path(root, name).with_extension(extension)
}

pub(crate) fn persist_collection(
    collection: &Collection,
    root: &Path,
) -> Result<(), VectorDbError> {
    let bytes = encode_collection(collection)?;
    let wal = wal_path(root, collection.name());
    let main_path = collection_path(root, collection.name());

    write_bytes_atomic(&wal, &bytes)?;
    write_bytes_atomic(&main_path, &bytes)?;

    match fs::remove_file(&wal) {
        Ok(_) => {}
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => {}
        Err(err) => return Err(VectorDbError::from(err)),
    }

    Ok(())
}

pub(crate) fn delete_collection_files(root: &Path, name: &str) -> Result<(), VectorDbError> {
    validate_collection_name(name)?;
    let main_path = collection_path(root, name);
    if !main_path.exists() {
        return Err(VectorDbError::CorruptedStorage(format!(
            "collection file '{}' missing during delete",
            main_path.display()
        )));
    }
    fs::remove_file(&main_path)?;

    let wal = wal_path(root, name);
    if wal.exists() {
        fs::remove_file(wal)?;
    }

    Ok(())
}

pub(crate) fn rename_collection_files(
    root: &Path,
    from: &str,
    to: &str,
) -> Result<(), VectorDbError> {
    validate_collection_name(from)?;
    validate_collection_name(to)?;
    let from_path = collection_path(root, from);
    if !from_path.exists() {
        return Err(VectorDbError::CorruptedStorage(format!(
            "collection file '{}' missing during rename",
            from_path.display()
        )));
    }
    let to_path = collection_path(root, to);
    if to_path.exists() {
        return Err(VectorDbError::CollectionAlreadyExists(to.to_string()));
    }

    fs::rename(&from_path, &to_path)?;

    let from_wal = wal_path(root, from);
    if from_wal.exists() {
        let to_wal = wal_path(root, to);
        if to_wal.exists() {
            return Err(VectorDbError::CollectionAlreadyExists(to.to_string()));
        }
        fs::rename(from_wal, to_wal)?;
    }

    Ok(())
}

pub(crate) fn load_collection(name: &str, path: &Path) -> Result<Collection, VectorDbError> {
    validate_collection_name(name)?;
    let bytes = fs::read(path)?;
    decode_collection(name, &bytes)
}

pub(crate) fn replay_wal(root: &Path) -> Result<(), VectorDbError> {
    for entry in fs::read_dir(root)? {
        let entry = entry?;
        let path = entry.path();
        if path.extension().and_then(|ext| ext.to_str()) != Some(WAL_SUFFIX) {
            continue;
        }

        let stem = path
            .file_stem()
            .and_then(|stem| stem.to_str())
            .ok_or_else(|| VectorDbError::CorruptedStorage("invalid wal filename".into()))?;
        let collection_name = Path::new(stem)
            .file_stem()
            .and_then(|stem| stem.to_str())
            .ok_or_else(|| VectorDbError::CorruptedStorage("invalid wal stem".into()))?;

        let bytes = fs::read(&path)?;
        let main_path = collection_path(root, collection_name);
        write_bytes_atomic(&main_path, &bytes)?;
        fs::remove_file(&path)?;
    }

    Ok(())
}

pub(crate) fn encode_collection(collection: &Collection) -> Result<Vec<u8>, VectorDbError> {
    let mut buffer = Vec::new();
    buffer.write_all(&FILE_MAGIC)?;
    write_u16(&mut buffer, FILE_VERSION)?;
    write_u64(&mut buffer, collection.dimension() as u64)?;
    write_u64(&mut buffer, collection.embeddings().len() as u64)?;

    for embedding in collection.embeddings() {
        write_u64(&mut buffer, embedding.id)?;
        write_u64(&mut buffer, embedding.vector.len() as u64)?;
        for value in &embedding.vector {
            write_f32(&mut buffer, *value)?;
        }
        write_u64(&mut buffer, embedding.metadata.len() as u64)?;
        for (key, value) in &embedding.metadata {
            write_string(&mut buffer, key)?;
            write_string(&mut buffer, value)?;
        }
    }

    Ok(buffer)
}

pub(crate) fn decode_collection(name: &str, bytes: &[u8]) -> Result<Collection, VectorDbError> {
    let mut cursor = Cursor::new(bytes);
    validate_file_header(&mut cursor)?;

    let dimension = read_u64(&mut cursor)?;
    let dimension = usize::try_from(dimension)
        .map_err(|_| VectorDbError::CorruptedStorage("dimension does not fit usize".into()))?;

    let entry_count = read_u64(&mut cursor)?;
    let entry_count = usize::try_from(entry_count)
        .map_err(|_| VectorDbError::CorruptedStorage("entry count does not fit usize".into()))?;

    let mut collection = Collection::new(name.to_string(), dimension)?;
    for _ in 0..entry_count {
        let id = read_u64(&mut cursor)?;
        let vector_len = read_u64(&mut cursor)?;
        let vector_len = usize::try_from(vector_len).map_err(|_| {
            VectorDbError::CorruptedStorage("vector length does not fit usize".into())
        })?;

        let mut vector = Vec::with_capacity(vector_len);
        for _ in 0..vector_len {
            vector.push(read_f32(&mut cursor)?);
        }

        let metadata_len = read_u64(&mut cursor)?;
        let metadata_len = usize::try_from(metadata_len).map_err(|_| {
            VectorDbError::CorruptedStorage("metadata length does not fit usize".into())
        })?;

        let mut metadata = Metadata::new();
        for _ in 0..metadata_len {
            let key = read_string(&mut cursor)?;
            let value = read_string(&mut cursor)?;
            metadata.insert(key, value);
        }

        collection
            .insert_embedding(Embedding::new(id, vector, metadata))
            .map_err(|err| {
                VectorDbError::CorruptedStorage(format!("failed to load embedding: {err}"))
            })?;
    }

    Ok(collection)
}

fn write_bytes_atomic(path: &Path, bytes: &[u8]) -> Result<(), VectorDbError> {
    let tmp_path = path.with_extension("tmp");
    {
        let mut file = File::create(&tmp_path)?;
        file.write_all(bytes)?;
        file.flush()?;
        file.sync_all()?;
    }

    match fs::rename(&tmp_path, path) {
        Ok(_) => Ok(()),
        Err(err) if err.kind() == std::io::ErrorKind::AlreadyExists => {
            fs::remove_file(path)?;
            fs::rename(&tmp_path, path)?;
            Ok(())
        }
        Err(err) => {
            let _ = fs::remove_file(&tmp_path);
            Err(VectorDbError::from(err))
        }
    }
}

fn validate_file_header<R: Read>(reader: &mut R) -> Result<(), VectorDbError> {
    let mut magic = [0u8; 4];
    reader.read_exact(&mut magic)?;
    if magic != FILE_MAGIC {
        return Err(VectorDbError::CorruptedStorage("invalid file magic".into()));
    }
    let version = read_u16(reader)?;
    if version != FILE_VERSION {
        return Err(VectorDbError::CorruptedStorage(format!(
            "unsupported file version {version}"
        )));
    }
    Ok(())
}

fn write_u16<W: Write>(writer: &mut W, value: u16) -> Result<(), VectorDbError> {
    writer.write_all(&value.to_le_bytes())?;
    Ok(())
}

fn write_u64<W: Write>(writer: &mut W, value: u64) -> Result<(), VectorDbError> {
    writer.write_all(&value.to_le_bytes())?;
    Ok(())
}

fn write_f32<W: Write>(writer: &mut W, value: f32) -> Result<(), VectorDbError> {
    writer.write_all(&value.to_bits().to_le_bytes())?;
    Ok(())
}

fn write_string<W: Write>(writer: &mut W, value: &str) -> Result<(), VectorDbError> {
    let bytes = value.as_bytes();
    write_u64(writer, bytes.len() as u64)?;
    writer.write_all(bytes)?;
    Ok(())
}

fn read_u16<R: Read>(reader: &mut R) -> Result<u16, VectorDbError> {
    let mut buf = [0u8; 2];
    reader.read_exact(&mut buf)?;
    Ok(u16::from_le_bytes(buf))
}

fn read_u64<R: Read>(reader: &mut R) -> Result<u64, VectorDbError> {
    let mut buf = [0u8; 8];
    reader.read_exact(&mut buf)?;
    Ok(u64::from_le_bytes(buf))
}

fn read_f32<R: Read>(reader: &mut R) -> Result<f32, VectorDbError> {
    let mut buf = [0u8; 4];
    reader.read_exact(&mut buf)?;
    Ok(f32::from_le_bytes(buf))
}

fn read_string<R: Read>(reader: &mut R) -> Result<String, VectorDbError> {
    let len = read_u64(reader)?;
    let len = usize::try_from(len)
        .map_err(|_| VectorDbError::CorruptedStorage("string length does not fit usize".into()))?;
    let mut buf = vec![0u8; len];
    reader.read_exact(&mut buf)?;
    String::from_utf8(buf).map_err(|err| VectorDbError::CorruptedStorage(err.to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::embedding::Metadata;

    #[test]
    fn encode_and_decode_round_trip() {
        let mut collection = Collection::new("docs".into(), 2).unwrap();
        let mut metadata = Metadata::new();
        metadata.insert("title".into(), "Doc".into());
        collection
            .insert_embedding(Embedding::new(1, vec![0.1, 0.2], metadata))
            .unwrap();

        let bytes = encode_collection(&collection).unwrap();
        let decoded = decode_collection("docs", &bytes).unwrap();

        assert_eq!(decoded.dimension(), 2);
        assert!(decoded.embedding(1).is_some());
    }
}
