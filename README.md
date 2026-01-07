# Vector DB

A minimal on-disk vector database with a flat index and an optional HTTP API server.

## Features

- Persistent collections of fixed-dimension embeddings
- Flat Euclidean search with deterministic ordering
- JSON HTTP API built with Axum, including structured errors and optional bearer authentication
- CLI configuration via flags or environment variables

## Running the HTTP Server

Build and launch the server binary with a storage directory:

```bash
cargo run --bin server -- --storage ./data --bind 0.0.0.0:8080
```

Environment variables can be used instead of CLI flags:

| Variable | Description | Default |
|----------|-------------|---------|
| `VECTORDB_STORAGE` | Path to the storage directory | `./vectordb-data` |
| `VECTORDB_BIND` | Socket address to listen on | `127.0.0.1:8080` |
| `VECTORDB_AUTH_TOKEN` | Optional static bearer token required on requests | unset |
| `VECTORDB_LOG` | Minimum log level (`error`, `warn`, `info`, `debug`, `trace`) | `info` |

The server exposes a health check at `GET /healthz`.

## Example Workflow

```bash
# Create a collection
curl -X POST http://localhost:8080/collections \
  -H 'Content-Type: application/json' \
  -d '{"name":"documents","dimension":3}'

# Insert an embedding
curl -X POST http://localhost:8080/collections/documents/vectors \
  -H 'Content-Type: application/json' \
  -d '{"id":1,"vector":[0.0,1.0,0.5],"metadata":{"title":"Doc A"}}'

# Update collection dimension
curl -X PUT http://localhost:8080/collections/documents \
  -H 'Content-Type: application/json' \
  -d '{"dimension":3}'

# Rename a collection
curl -X POST http://localhost:8080/collections/documents/rename \
  -H 'Content-Type: application/json' \
  -d '{"name":"docs"}'

# Perform a search
curl -X POST http://localhost:8080/collections/docs/search \
  -H 'Content-Type: application/json' \
  -d '{"query":[0.0,1.0,0.4],"k":1}'

# Delete a collection
curl -X DELETE http://localhost:8080/collections/docs
```

When authentication is enabled set `Authorization: Bearer <token>` on modifying requests.

## API Documentation

OpenAPI documentation and usage examples are available under [`docs/`](docs/):

- [`docs/openapi.yaml`](docs/openapi.yaml) – OpenAPI 3.1 specification for the HTTP API.
- [`docs/usage.md`](docs/usage.md) – Additional examples and workflow tips.

## Testing

Run the full test suite, including integration tests that exercise the HTTP server:

```bash
cargo test
```
