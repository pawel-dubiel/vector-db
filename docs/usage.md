# Usage Examples

## Creating and Listing Collections

```bash
curl -s http://localhost:8080/collections | jq

curl -X POST http://localhost:8080/collections \
  -H 'Content-Type: application/json' \
  -d '{"name":"images","dimension":512}'
```

## Updating, Renaming, and Deleting Collections

```bash
curl -X PUT http://localhost:8080/collections/images \
  -H 'Content-Type: application/json' \
  -d '{"dimension":256}'

curl -X POST http://localhost:8080/collections/images/rename \
  -H 'Content-Type: application/json' \
  -d '{"name":"pictures"}'

curl -X DELETE http://localhost:8080/collections/pictures
```

## Working with Embeddings

```bash
curl -X POST http://localhost:8080/collections/images/vectors \
  -H 'Content-Type: application/json' \
  -d '{"id":42,"vector":[0.1,0.2,0.3,0.4],"metadata":{"label":"cat"}}'

curl http://localhost:8080/collections/images/vectors/42
```

## Performing Searches

```bash
curl -X POST http://localhost:8080/collections/images/search \
  -H 'Content-Type: application/json' \
  -d '{"query":[0.1,0.2,0.3,0.4],"k":5}'
```

## Using Authentication

```bash
export VECTORDB_AUTH_TOKEN=secret
cargo run --bin server -- --storage ./data

curl -X POST http://localhost:8080/collections \
  -H 'Authorization: Bearer secret' \
  -H 'Content-Type: application/json' \
  -d '{"name":"secure","dimension":4}'
```
