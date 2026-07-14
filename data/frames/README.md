---
license: apache-2.0
language:
- en
tags:
- rag
- long-context
- llm-search
- reasoning
- factuality
- retrieval
- question-answering
- iterative-search
task_categories:
- text-classification
- token-classification
- table-question-answering
- question-answering
pretty_name: Who are I or you
size_categories:
- n>1T
---

# FRAMES: Factuality, Retrieval, And reasoning MEasurement Set

FRAMES is a comprehensive evaluation dataset designed to test the capabilities of Retrieval-Augmented Generation (RAG) systems across factuality, retrieval accuracy, and reasoning.
Our paper with details and experiments is available on arXiv: [https://arxiv.org/abs/2409.12941](https://arxiv.org/abs/2409.12941).


## Dataset Overview

- 824 challenging multi-hop questions requiring information from 2-15 Wikipedia articles
- Questions span diverse topics including history, sports, science, animals, health, etc.
- Each question is labeled with reasoning types: numerical, tabular, multiple constraints, temporal, and post-processing
- Gold answers and relevant Wikipedia articles provided for each question

## Key Features

- Tests end-to-end RAG capabilities in a unified framework
- Requires integration of information from multiple sources
- Incorporates complex reasoning and temporal disambiguation
- Designed to be challenging for state-of-the-art language models

## Usage

This dataset can be used to:
- Evaluate RAG system performance 
- Benchmark language model factuality and reasoning
- Develop and test multi-hop retrieval strategies

## Baseline Results

We provide baseline results using state-of-the-art models like Gemini-Pro-1.5-0514:

- Naive prompting: 40.8% accuracy
- BM25 retrieval (4 docs): 47.4% accuracy  
- Oracle retrieval: 72.9% accuracy
- Multi-step retrieval & reasoning: 66% accuracy

## Citation

If you use this dataset in your research, please cite our paper:

```
@misc{krishna2024factfetchreasonunified,
      title={Fact, Fetch, and Reason: A Unified Evaluation of Retrieval-Augmented Generation}, 
      author={Satyapriya Krishna and Kalpesh Krishna and Anhad Mohananey and Steven Schwarcz and Adam Stambler and Shyam Upadhyay and Manaal Faruqui},
      year={2024},
      eprint={2409.12941},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2409.12941}, 
}
```

We hope FRAMES will be useful for advancing RAG systems and language model capabilities. For more details, please refer to our full paper.