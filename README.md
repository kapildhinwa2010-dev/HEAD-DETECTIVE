# HEAD-DETECTIVE

Classify what every attention head actually does, straight from its attention patterns.

Given attention matrices, HEAD-DETECTIVE computes per-head statistics (diagonal mass, previous-token mass, band mass, long-range mass, entropy, sink mass) and assigns each head a role label with a confidence score: **previous-token**, **local**, **diffuse**, **long-range**, **sink**, or **mixed**.

## What it does

- Classifies each attention head from pattern statistics alone.
- Works on any attention matrix, from any framework.
- Handles single heads, head stacks, and batched stacks.
- Returns a count summary across all heads.

## Install

```bash
pip install -e .
```

## Quick start

```python
import numpy as np
from head_detective.detective import detective_report, summarize

# patterns: (heads, T, T) attention matrices, rows are queries
patterns = np.load("patterns.npy")
report = detective_report(patterns)
for head in report:
    print(head["role"], head["confidence"])
print(summarize(report))
```

## How it works

Each pattern is row-normalized, then measured on five axes:

- attention mass on the diagonal and nearby band
- attention mass on the previous position
- attention mass beyond a quarter of the sequence
- normalized row entropy
- concentration on a single key position (sink behavior)

The first axis that crosses its threshold wins, in a fixed order.

## Honest scope

- Role labels are heuristics from pattern statistics, not causal claims.
- The core is pure NumPy; extract patterns from your model with your own tooling.
- Tests use hand-built synthetic patterns, so no model download is needed.

## License

MIT