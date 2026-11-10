# Superseded holdout runs of the LLM systems

The holdout investigations in `results/genai/holdout/` (version C) are the final results used in the
report. Before them, the ML + LLM and gated systems were run twice more.

| Version | Saved | What differed from the final run |
|---|---|---|
| `A_before_final_regressor`  the blast-radius forecast came from the regressor used before the final XGBoost one |
| `B_before_clip_at_zero`  | negative blast-radius forecasts were shown to the agent as they were, not as 0 |
| final (`results/genai/holdout/`)  | |

In all three versions the prompt text, the ML severity forecast and its probabilities, and the incidents sent to the
agent by the gate (38 of 50) are the same. Only the blast-radius number in the ML forecast shown to the agent differs,
by at most 0.2 services in the saved prompts. The LLM-only system does not see the ML forecast and was not re-run, so
its rows are the same in all versions. When the forecast changed, the investigations whose prompt changed were re-run
and the result files overwritten; the earlier files were backed up and are copied here.

Contents, for each model and repeat: `results.csv` (all systems, as saved at the time) and the `prompt.txt` of the
investigations that were backed up (not all of them; the agents' full transcripts are not included).

Severity accuracy of the gated system with gpt-6-sol (mean of three repeats): A 0.833, B 0.867, final 0.853.
Reproduce with:

```python
import glob, pandas as pd
for v in ("A_before_final_regressor", "B_before_clip_at_zero"):
    d = pd.concat(pd.read_csv(f) for f in glob.glob(f"results/genai/superseded/{v}/gpt-6-sol/rep*/results.csv"))
    print(v, round(d[d.system == "gated"].severity_correct.mean(), 3))
```
