# Heuristics

## H01: Do not evaluate AF only through final answer correctness
- **Rationale**: AF may help by making claims, conflicts, accepted/rejected status, and evidence provenance inspectable even when final-answer QA metrics show little gain.
- **Provenance**: ai-suggested
- **Sensitivity**: medium
- **Code ref**: [`OASIS/ablation-AF/experiment.py`, `OASIS/ablation-AF/research_2025_plus.md`]

## H02: Keep all graph statuses visible for coverage tasks
- **Rationale**: DebateQA rewards coverage of legitimate partial answers, while grounded AF semantics can compress content into accepted arguments and suppress rejected or undecided perspectives.
- **Provenance**: ai-suggested
- **Sensitivity**: high
- **Code ref**: [`OASIS/ablation-AF/prompts.py`, `OASIS/ablation-AF/af_pipeline.py`]

## H03: Select datasets by downstream decision label
- **Rationale**: To isolate the value of AF graph integration, the benchmark should contain a decision label after long natural-language evidence or generated interaction traces. Argument mining can be a pipeline stage, but not the primary metric.
- **Provenance**: user
- **Sensitivity**: high
- **Code ref**: [`OASIS/ablation-AF/af_decision_benchmark_recommendations.md`]

## H04: Score categorical decisions conservatively
- **Rationale**: Short labels such as `B`, `CG`, `accept`, or `faithful` can appear inside unrelated generic answers, so categorical one-token contains matches should only count when the extracted decision answer is short and label-like.
- **Provenance**: ai-suggested
- **Sensitivity**: medium
- **Code ref**: [`OASIS/ablation-AF/decision_common.py`]

## H05: Use semantic equivalence for memory-answer evaluation
- **Rationale**: LongMemEval-style answers often paraphrase the same remembered event, so exact/contains scoring can undercount correct outputs such as "GPS system issue" versus "GPS system not functioning correctly".
- **Provenance**: ai-suggested
- **Sensitivity**: high
- **Code ref**: [`OASIS/ablation-AF/memory_longmemeval_v2/run_longmemeval_v2.py`, `OASIS/ablation-AF/outputs/longmemeval_v2_decision.jsonl`]
