```
srun --gres=gpu:4 --qos=high --mem=32G bash -c "source .venv/bin/activate && torchrun --nproc_per_node=4 moe_wikitext_lplb_demo.py --steps 100 --profile-warmup=5 --profile-interval=1 --redundants-per-rank=4
```

200 

1
[profile aggregate]
  steps profiled: 98
  total time: 1279.00ms/step
  data loading: 1.46ms
  refresh mapping: 0.00ms
  forward pass: 448.46ms
  backward pass: 792.47ms
  optimizer step: 29.35ms
  throughput: 3202.51 tokens/sec

2
[profile aggregate]
  steps profiled: 98
  total time: 1455.32ms/step
  data loading: 24.12ms
  refresh mapping: 20.20ms
  forward pass: 589.72ms
  backward pass: 810.62ms
  optimizer step: 23.06ms
  throughput: 2814.51 tokens/sec

## hello world

info for 1 redundants
[profile aggregate]
  steps profiled: 95
  total time: 2517.32ms/step 2549
  data loading: 17.79ms
  refresh mapping: 16.74ms
  forward pass: 2434.90ms
  aux metric: 12.007908
  backward pass: 62.49ms
  optimizer step: 1.65ms
  router kernel (self.router): 0.00µs
  planner.run: 0.00µs
  router topk+weights: 1.32ms
  router assignment: 185.92ms
  router dispatch prep: 0.58ms
  router total: 187.82ms
  throughput: 1627.12 tokens/sec

info for 2 redundants
[profile aggregate]
  steps profiled: 95
  total time: 2450.53ms/step 2480ms/step
  data loading: 17.48ms
  refresh mapping: 16.61ms
  forward pass: 2368.63ms
  aux metric: 12.682679
  backward pass: 62.26ms
  optimizer step: 1.68ms
  router kernel (self.router): 0.00µs
  planner.run: 0.00µs
  router topk+weights: 1.70ms
  router assignment: 186.01ms
  router dispatch prep: 0.61ms
  router total: 188.32ms
  throughput: 1671.47 tokens/sec


info for 2 redundant with high lplb threshold
[profile aggregate]
  steps profiled: 85
  total time: 2425.95ms/step
  data loading: 19.76ms
  refresh mapping: 17.90ms
  forward pass: 2339.23ms
  aux metric: 13.576462
  backward pass: 62.45ms
  optimizer step: 4.08ms
  router kernel (self.router): 0.00µs
  planner.run: 0.00µs
  router topk+weights: 1.65ms
  router assignment: 210.96ms
  router dispatch prep: 0.60ms
  router total: 213.21ms
  throughput: 1688.41 tokens/sec

info for 1 redundant with high lplb threshold
[profile aggregate]
  steps profiled: 85
  total time: 2563.09ms/step
  data loading: 18.09ms
  refresh mapping: 16.40ms
  forward pass: 2478.36ms
  aux metric: 12.906646
  backward pass: 63.59ms
  optimizer step: 2.64ms
  router kernel (self.router): 0.00µs
  planner.run: 0.00µs
  router topk+weights: 1.76ms
  router assignment: 217.62ms
  router dispatch prep: 0.61ms
  router total: 219.99ms
  throughput: 1598.07 tokens/sec