# Relation Extraction

Extract relations and build knowledge graphs from text and entities. Discovers relationships, resolves duplicate entities, and generates structured graph output.

```python
from latence import Latence
client = Latence(api_key="lat_xxx")

result = client.experimental.ontology.build_graph(
    text="Elon Musk founded SpaceX in Hawthorne, California.",
    entities=[],  # Empty = extract entities first
)
print(f"{result.entity_count} entities, {result.relation_count} relations")
for rel in result.relations:
    print(f"{rel.source_entity} --[{rel.relation_type}]--> {rel.target_entity}")
```

> **Note:** Direct service APIs live under `client.experimental.*`. For production workloads, prefer the [pipeline](pipelines.md) which includes `relation_extraction` (aliases: `ontology`, `knowledge_graph`) as a pipeline step.

## `client.experimental.ontology.build_graph()`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `text` | `str` | required | Source text (at least one of `text`/`entities` required) |
| `entities` | `list[dict] \| list[EntityInput]` | required | Pre-extracted entities (can be empty) |
| `config` | `dict \| OntologyConfig \| None` | `None` | Graph construction configuration |
| `request_id` | `str \| None` | `None` | Optional tracking ID |
| `return_job` | `bool` | `False` | Return job ID for async polling |

### Config Options (`OntologyConfig`)

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `relation_threshold` | `float` | `0.6` | Minimum confidence for relations (0.0-1.0) |
| `symmetric` | `bool` | `True` | Create bidirectional relations |
| `generate_knowledge_graph` | `bool` | `True` | Generate structured KG output |
| `max_relations_per_decode` | `int` | `30` | Max relations per batch decode |
| `resolve_entities` | `bool` | `True` | Merge duplicate entities |
| `optimize_relations` | `bool` | `True` | Refine relation labels with AI |
| `optimize_entity_resolution` | `bool` | `False` | Optimize entity resolution |
| `predict_missing_relations` | `bool` | `False` | Predict unobserved relations |
| `link_prediction_verify_with_ai` | `bool` | `False` | AI verification for predicted links |
| `kg_output_format` | `"custom" \| "rdf" \| "property_graph"` | `"custom"` | Knowledge graph format |
| `namespace_uri` | `str \| None` | `None` | RDF namespace URI (for `rdf` format) |

### Entity Input Format

```python
entities = [
    {"text": "Elon Musk", "label": "PERSON", "start": 0, "end": 9, "score": 0.98},
    {"text": "SpaceX", "label": "ORGANIZATION", "start": 18, "end": 24, "score": 0.97},
]
```

### Using with Entity Extraction

```python
# Step 1: Extract entities
extraction = client.experimental.extraction.extract(text=text)

# Step 2: Build graph from extracted entities
graph = client.experimental.ontology.build_graph(
    text=text,
    entities=[e.model_dump() for e in extraction.entities],
)
```

## Response: `BuildGraphResponse`

| Field | Type | Description |
|-------|------|-------------|
| `entities` | `list[Entity]` | Processed entities |
| `relations` | `list[OntologyRelation]` | Discovered relations |
| `knowledge_graph` | `KnowledgeGraph \| str \| None` | Structured graph (dict for `custom`/`property_graph`, RDF string for `rdf`) |
| `entity_count` | `int` | Number of entities |
| `relation_count` | `int` | Number of relations |
| `resolved_entities` | `int \| None` | Entities after deduplication |
| `processing_stats` | `dict \| None` | Processing statistics |
| `usage` | `Usage \| None` | Credits consumed |

### Relation Fields

Each `OntologyRelation` provides:
- `entity1`, `entity2` -- the related entities
- `score` -- confidence (0-1)
- `relation_type` -- relation label (property)
- `source_entity`, `target_entity` -- text shortcuts (properties)

## Pricing

| Component | Cost |
|-----------|------|
| Base | $20.00 / 1M tokens |
| Relation optimization (`optimize_relations=True`) | +$25.00 / 1M tokens |
| Entity resolution (`resolve_entities=True`) | +$10.00 / 1M tokens |
| Link prediction (`predict_missing_relations=True`) | +$50.00 / 1M tokens |
