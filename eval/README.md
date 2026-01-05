# Eval Harness

## Dataset format

Provide a JSON list (or JSONL) of objects with:
- `query` (string, required)
- `expected_decision` (`supported` or `not_supported`, required)
- `expected_pages` (optional; not enforced yet, reserved for future)

Example: `eval/dataset.example.json`

## Run

```bash
./.venv/bin/python -m eval.run --pdf knowledge/dspy.pdf --dataset eval/dataset.example.json
```

