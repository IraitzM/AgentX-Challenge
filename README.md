# Finance agent

This works takes the benchmark in https://arxiv.org/abs/2508.00828
The original code for the dataset is in https://github.com/vals-ai/finance-agent

![](https://www.vals.ai/_astro/slide_LLM.wr9G0gyQ_ZnttS3.webp)

## Configuration

Currently uses Nebius by default and Kimi's K2 Instruct model. Make sure you add a local _.env_ file following the example in _.env.example_.

### Log Levels

The default log level is set to `INFO`. To change the log level, set the `LOG_LEVEL` environment variable:

```sh
# Set log level to DEBUG to see detailed rubric evaluation information
export LOG_LEVEL=DEBUG
uv run finben run

# Or set it inline
LOG_LEVEL=DEBUG uv run finben run
```

Available log levels (from most verbose to least):
- `DEBUG` - Shows detailed information including rubric evaluation details (question, answer, criteria, scores)
- `INFO` - Shows general progress and important messages (default)
- `WARNING` - Shows only warnings
- `ERROR` - Shows only errors

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
- [x] Finish rubric evaluation
- [ ] Wrap it on an accesible machine https://docs.agentbeats.org/Blogs/blog-3/#agentbeats-controller

NOTE: Last point might not be needed to register the agent in the agentbeats.dev platform.