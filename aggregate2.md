warmup then 80 trials

normal EPLB
[profile aggregate]
  steps profiled: 85
  total time: 2407.61ms/step
  data loading: 16.48ms
  refresh placement: 15.20ms
  forward pass: 2312.96ms
  aux metric: 12.689436
  backward pass: 75.08ms
  optimizer step: 2.20ms
  router kernel (self.router): 0.00µs
  eplb_refresh: 0.00µs
  router topk+weights: 2.82ms
  router assignment: 1.32ms
  router dispatch prep: 0.85ms
  router total: 4.99ms
  throughput: 1701.27 tokens/sec

using the overlap informed EPLB
[profile aggregate]
  steps profiled: 80
  total time: 2373.53ms/step
  data loading: 17.42ms
  refresh placement: 15.87ms
  forward pass: 2278.45ms
  aux metric: 13.193908
  backward pass: 74.70ms
  optimizer step: 2.28ms
  router kernel (self.router): 0.00µs
  eplb_refresh: 0.00µs
  router topk+weights: 2.87ms
  router assignment: 1.36ms
  router dispatch prep: 0.96ms
  router total: 5.19ms
  throughput: 1725.70 tokens/sec