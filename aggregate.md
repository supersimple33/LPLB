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



info 2 redundants with parallel dispatch
[profile aggregate]
  steps profiled: 95
  total time: 2611.71ms/step 4123ms/step
  data loading: 17.80ms
  refresh mapping: 16.46ms
  forward pass: 2526.46ms
  aux metric: 12.679115
  backward pass: 64.75ms
  optimizer step: 2.20ms
  router kernel (self.router): 0.00µs
  planner.run: 0.00µs
  router topk+weights: 1.58ms
  router assignment: 193.23ms
  router dispatch prep: 0.60ms
  router total: 195.41ms
  throughput: 1568.32 tokens/sec

info 2 redundants with parallel dispatch and async comm
[profile aggregate]
  steps profiled: 85
  total time: 3889.62ms/step
  data loading: 19.22ms
  refresh mapping: 16.79ms
  forward pass: 3467.76ms
  aux metric: 17.639570
  backward pass: 393.74ms
  optimizer step: 7.41ms
  router kernel (self.router): 0.00µs
  planner.run: 0.00µs
  router topk+weights: 1.67ms
  router assignment: 214.99ms
  router dispatch prep: 0.65ms
  router total: 217.31ms
  throughput: 1053.06 tokens/sec

info 1 redundants with parallel dispatch and async comm
[profile aggregate]
  steps profiled: 85
  total time: 3899.59ms/step
  data loading: 17.85ms
  refresh mapping: 15.54ms
  forward pass: 3496.23ms
  aux metric: 17.659202
  backward pass: 377.51ms
  optimizer step: 6.79ms
  router kernel (self.router): 0.00µs
  planner.run: 0.00µs
  router topk+weights: 1.71ms
  router assignment: 213.52ms
  router dispatch prep: 0.61ms
  router total: 215.83ms
  throughput: 1050.37 tokens/sec