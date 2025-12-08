# Finance agent

This works takes the benchmark in https://arxiv.org/abs/2508.00828
The original code for the dataset is in https://github.com/vals-ai/finance-agent

![](https://www.vals.ai/_astro/slide_LLM.wr9G0gyQ_ZnttS3.webp)

## Running the benchmark

If uv is available, simply run
```sh
uv run finben run
```

For the agentbeats controller deployment in https://docs.agentbeats.org/Blogs/blog-3/#agentbeats-controller
```sh
agentbeats run_ctrl
```

## Pending

- [x] Indexed task
- [x] Green agent evaluator
- [x] White/Purple agent execution
- [ ] Finish rubric evaluation
- [ ] Wrap it on an accesible machine https://docs.agentbeats.org/Blogs/blog-3/#agentbeats-controller