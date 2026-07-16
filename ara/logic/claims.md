# Claims

## C01: DebateQA plus DeepSeek multi-round is a weak primary AF validation task
- **Statement**: On the inspected DebateQA dev20 DeepSeek runs, the `multi_round` arm outperforms the current `af_graph` arm on official P.D. and local perspective coverage, so this setup is weak evidence for AF graph usefulness.
- **Status**: supported
- **Provenance**: user
- **Falsification criteria**: A larger controlled DebateQA run with the same strong baseline shows `af_graph` or `af_graph_with_citations` reliably beating `multi_round` on official P.D. without losing D.A. or coverage.
- **Proof**: [`OASIS/ablation-AF/outputs/debateqa_dev20_llmmerge_deepseek_real_summary.md`, `OASIS/ablation-AF/outputs/debateqa_dev20_promptv2_deepseek_real_summary.md`]
- **Dependencies**: []
- **Tags**: AF, DebateQA, ablation, task-fit

## C02: AF should be tested where structure is the task signal
- **Statement**: AF graph extraction is more likely to show value on long-text, multi-turn, relation-aware, citation-grounded, or memory-management tasks than on already concise DebateQA answers.
- **Status**: hypothesis
- **Provenance**: user
- **Falsification criteria**: Structured long-context or memory tasks show no improvement from AF graph variants over raw-text and multi-round baselines across answer, citation, and graph-quality metrics.
- **Proof**: [pending]
- **Dependencies**: [C01]
- **Tags**: long-context, memory, citation, graph-quality

## C03: The validation target is graph-conditioned decision, not argument mining
- **Statement**: OASIS should evaluate whether an already constructed or system-constructed AF graph improves downstream decisions, rather than treating argument mining quality as the central task.
- **Status**: supported
- **Provenance**: user
- **Falsification criteria**: A pure argument-mining dataset directly evaluates the claimed OASIS contribution without any downstream decision task or graph-conditioned reasoning step.
- **Proof**: [`OASIS/ablation-AF/af_decision_benchmark_recommendations.md`]
- **Dependencies**: [C01, C02]
- **Tags**: AF, decision, benchmark-design

## C04: Peer review, RAG verification, memory, and generated-agent transcripts are stronger benchmark families
- **Statement**: For AF-after-construction decision validation, peer review decisions, RAG/claim verification, long-term memory/state tasks, and generated multi-agent interaction tasks better match the OASIS contribution than KIALO argument mining alone.
- **Status**: revised
- **Provenance**: ai-suggested
- **Falsification criteria**: These task families fail to provide reliable downstream labels, or AF graph arms cannot be evaluated independently from extraction quality.
- **Proof**: [`OASIS/ablation-AF/af_decision_benchmark_recommendations.md`]
- **Dependencies**: [C03]
- **Tags**: benchmark-selection, peer-review, RAG, memory, multi-agent

## C05: Peer review ratings are not the cleanest zero-shot decision benchmark
- **Statement**: Peer review datasets with rating or meta-review targets are less suitable as primary LLM decision benchmarks because the labels are subjective and often require reviewer-confidence or argument-strength modeling beyond basic AF.
- **Status**: supported
- **Provenance**: user
- **Falsification criteria**: A peer review task provides objective, reproducible, non-rating decision labels and shows that basic AF is sufficient without confidence/strength modeling.
- **Proof**: [`OASIS/ablation-AF/af_decision_benchmark_recommendations.md`]
- **Dependencies**: [C03]
- **Tags**: peer-review, QAF, QBAF, benchmark-selection

## C06: Court and public debate data are stronger primary AF decision targets
- **Statement**: Courtroom/legal-argument data and public debate transcripts better match OASIS than peer review ratings because they contain explicit adversarial structure and downstream decisions such as verdicts, winners, audience shifts, or legal argument choices.
- **Status**: hypothesis
- **Provenance**: user
- **Falsification criteria**: Legal/debate datasets lack usable downstream labels, or AF/QBAF formalization does not improve over raw transcript and summary baselines.
- **Proof**: [`OASIS/ablation-AF/af_decision_benchmark_recommendations.md`]
- **Dependencies**: [C03, C05]
- **Tags**: legal, debate, AF, QBAF, decision

## C07: OASIS now has modular decision-benchmark adapters
- **Statement**: The ablation-AF workspace now contains separate runnable adapters for LAR-ECHR, DebateBench, PeerRead, RAGTruth, LongMemEval-V2, and MALLM GPQA/MMLU-Pro decision tests, all sharing a common AF decision runner.
- **Status**: supported
- **Provenance**: ai-executed
- **Falsification criteria**: Any required dataset folder or runner is missing, syntax checks fail, or sample mock runs cannot produce row, graph, prompt, and summary outputs.
- **Proof**: [`OASIS/ablation-AF/decision_common.py`, `OASIS/ablation-AF/legal_lar_echr/`, `OASIS/ablation-AF/debatebench/`, `OASIS/ablation-AF/review_peerread/`, `OASIS/ablation-AF/ragtruth/`, `OASIS/ablation-AF/memory_longmemeval_v2/`, `OASIS/ablation-AF/mallm_gpqa_mmlu_pro/`]
- **Dependencies**: [C03, C06]
- **Tags**: implementation, adapters, AF, decision

## C08: The decision adapters can compare direct, AF, QBAF, and direct multi-agent baselines
- **Statement**: The shared decision runner now supports `direct_answer`, `arguments_only`, `af_graph`, `qbaf_graph`, and `multi_agent` arms, defaults decision adapters to `deepseek-chat`, emits per-folder `test_report.md` files, and has a suite runner that executes all six adapters with one configuration and aggregates per-arm deltas against `direct_answer`.
- **Status**: supported
- **Provenance**: ai-executed
- **Falsification criteria**: A decision adapter cannot run these arms, fails to emit row/summary/graph/prompt outputs and a folder report, or a real DeepSeek API run fails after a valid key and dataset input are provided.
- **Proof**: [`OASIS/ablation-AF/decision_common.py`, `OASIS/ablation-AF/llm_clients.py`, `OASIS/ablation-AF/credential_discovery.py`, `OASIS/ablation-AF/run_decision_suite.py`, `OASIS/ablation-AF/decision_suite_report.md`, `OASIS/ablation-AF/*/test_report.md`]
- **Dependencies**: [C07]
- **Tags**: implementation, DeepSeek, baseline, multi-agent, reporting

## C09: Raw long-text data is a better primary AF validation target than pre-structured argument data
- **Statement**: For the current OASIS AF validation, raw legal transcripts, raw online discussion threads, and raw public debates are better primary tests than datasets whose inputs are already segmented into argument-like units.
- **Status**: supported
- **Provenance**: user
- **Falsification criteria**: A controlled run shows that raw long-text datasets do not require formalization, while pre-structured datasets reliably show AF gains over direct and multi-round baselines.
- **Proof**: [`OASIS/ablation-AF/af_decision_benchmark_recommendations.md`, `OASIS/ablation-AF/legal_scotus_raw/`, `OASIS/ablation-AF/online_cmv_raw/`, `OASIS/ablation-AF/public_iq2_raw/`]
- **Dependencies**: [C01, C02, C06]
- **Tags**: raw-text, legal, online-discussion, debate, benchmark-selection

## C10: LongMemEval-V2 should formalize all memory before answering
- **Statement**: In the OASIS LongMemEval-V2 adapter, graph arms should extract structure from all rendered memory sessions before answering, not consume one pre-baked argument per memory item.
- **Status**: supported
- **Provenance**: user
- **Falsification criteria**: A memory benchmark demonstrates that pre-making one node per memory is equivalent to or better than full-session formalization across temporal reasoning, correction, and supersession questions.
- **Proof**: [`OASIS/ablation-AF/memory_longmemeval_v2/run_longmemeval_v2.py`, `OASIS/ablation-AF/memory_longmemeval_v2/README.md`, `OASIS/ablation-AF/outputs/longmemeval_v2_decision_graphs.jsonl`]
- **Dependencies**: [C02]
- **Tags**: LongMemEval, memory, formalization, context-management

## C11: MALLM blackboard state is a plausible AF formalization target
- **Statement**: A fresh controller + blackboard multi-agent QA episode gives AF formalization a more suitable input than pre-existing compact MALLM transcripts, because the graph can be built over accumulated agent state before final decision.
- **Status**: weakened
- **Provenance**: ai-suggested
- **Falsification criteria**: Larger GPQA/MMLU-Pro blackboard runs show no improvement in exact-label quality, graph diagnostics, or robustness over direct blackboard judging, and relation noise dominates the final decisions.
- **Proof**: [`OASIS/ablation-AF/mallm_gpqa_mmlu_pro/run_mallm_blackboard.py`, `OASIS/ablation-AF/outputs/mallm_blackboard_decision_real_50_summary.json`, `OASIS/ablation-AF/decision_suite_mallm_blackboard_real_50.md`]
- **Dependencies**: [C02, C03, C10]
- **Tags**: MALLM, blackboard, multi-agent, AF, context-management

## C12: LongMemEval exact-string metrics understate semantic correctness
- **Statement**: The current LongMemEval-V2 exact/contains metrics can mark semantically correct answers as wrong when the model paraphrases the memory fact.
- **Status**: supported
- **Provenance**: ai-suggested
- **Falsification criteria**: A semantic judge or expanded alias set agrees with exact/contains scoring on representative LongMemEval-V2 outputs, including paraphrased temporal-memory answers.
- **Proof**: [`OASIS/ablation-AF/outputs/longmemeval_v2_decision.jsonl`, `OASIS/ablation-AF/real_model_small_run_notes.md`]
- **Dependencies**: [C10]
- **Tags**: LongMemEval, evaluation, semantic-equivalence, metrics

## C13: Current AF graph decision gains are concentrated in LongMemEval memory formalization
- **Statement**: In the 50-example real-model run, AF graph decision prompting does not broadly outperform direct prompting; its only positive accuracy delta among the tested tasks is on LongMemEval-V2.
- **Status**: supported
- **Provenance**: ai-executed
- **Falsification criteria**: Larger or corrected runs show AF graph arms reliably beating direct prompting on raw legal/debate/forum tasks or MALLM blackboard QA without losing graph quality.
- **Proof**: [`OASIS/ablation-AF/real_model_50_run_notes.md`, `OASIS/ablation-AF/decision_suite_raw_real_50.md`, `OASIS/ablation-AF/decision_suite_mallm_blackboard_real_50.md`]
- **Dependencies**: [C09, C10, C11]
- **Tags**: AF, LongMemEval, memory, benchmark-results, DeepSeek
