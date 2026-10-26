# Output Format

Each run writes a single output directory:

```text
outputs/<run_name>/
  run_config.json
  input_manifest.csv
  metrics/<metric_name>/
    per_sample.csv
    model_summary.csv
    ranking.md
    details.json
```

`per_sample.csv` uses stable common fields:

```text
sample_id, model_name, video_path, metric_name, score, status, reason, raw_score
```

Metric-specific diagnostic fields may be appended.

