# Project proposal – ICL example generation.

**Keywords:** In-Context Learning, Synthetic Data, Genetic Algorithms, Prompt Engineering

Large Language Models (LLMs) are powerful across many domains, but applying them to narrow, domain‑specific tasks is challenging when the required knowledge isn’t present in their parametric memory. Instead of costly supervised finetuning, we can leverage instruction-following by supplying high‑quality, domain‑specific In‑Context Learning (ICL) examples. However, crafting good examples that capture domain nuances is intricate and error‑prone. 

## Objective
Given a domain corpus, automatically generate and optimize ICL example sets that improve LLM performance on domain‑specific tasks. The work targets high practical impact and aims to lead to a scientific publication. 

## Approach
We implement and evaluate a three‑step pipeline: 

1. **Over-generate candidates:** Use LLMs to produce many synthetic candidate ICL examples from the domain corpus.   
2. **Cluster & select for coverage:** Identify themes/clusters and select a small number of diverse candidates from each cluster to reduce total examples while maintaining domain coverage.   
3. **Optimize with GA:** Use genetic algorithms (or other optimization techniques) to search for the optimal set of examples to include in prompts. 

## Why This Matters
Automating ICL example generation and selection lowers the cost of adapting LLMs to specialized domains and reduces manual prompt‑engineering effort. 
