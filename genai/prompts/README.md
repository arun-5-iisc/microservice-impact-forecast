# System prompts

There are two versions of the agent's system prompt:

| Version | What it is |
|---|---|
| `system_v0.md` | The first prompt: the task, the severity definition, and the rule to base every conclusion on observed evidence and lower the confidence when the evidence is weak. |
| `system_v1.md` | v0 plus one paragraph about rollouts: a change often replaces the pod, brief failures while callers move to the new pod usually clear within a minute, so the final state should be judged from the new pod's behaviour. It was added after development round 0, where 6 of the 7 false alerts were CPU/memory rollouts. **This is the version used for the holdout evaluation** (`FROZEN_PROMPT` in `genai/agent.py`). |

The prompt was refined on a separate **development set**: incidents from the training runs only, so the holdout
incidents are never looked at while the prompt changes. The set is stratified so that every cause and final severity
is covered (at least 2 incidents each where there are that many; `genai/dev_set.py`). 

## How to run

The version is chosen with `--prompt <version>` in `genai/orchestrator.py`; `scripts/run_genai_eval.sh` picks the
right one for each stage:

```bash
./scripts/run_genai_eval.sh prepare                                         # once: span cache, RAG index, development set
./scripts/run_genai_eval.sh dev 0 --yes                                     # development round 0, prompt v0
./scripts/run_genai_eval.sh dev 1 --yes                                     # development round 1, prompt v1
REPEATS=3 MODEL=claude-opus-5-5 ./scripts/run_genai_eval.sh holdout --yes   # holdout, prompt v1
MODEL=claude-opus-5-5 ./scripts/run_genai_eval.sh sensitivity --yes         # llm_only on the holdout with prompt v0
```

- Claude models need `ANTHROPIC_API_KEY` or `~/.anthropic_key`; OpenAI models (`MODEL=gpt-6-sol`) need
  `OPENAI_API_KEY` or `~/.openai_key`.
- `PROMPT=<version>` overrides the prompt of a stage, and `SENS_PROMPT=<version>` the one the sensitivity stage
  compares against (default v0).
- Every results row records the version (`prompt`) and a short fingerprint of its text (`prompt_sha`: v0
  `a07fbe16349d`, v1 `ff4124adec09`). The full prompt of each investigation is saved as `prompt.txt` next to its
  transcript.
- A results folder holds one prompt version only. Running a stage again resumes its folder instead of starting over.
