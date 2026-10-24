---
title: frugal-eval
emoji: 🎯
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 6.30.0
app_file: app.py
pinned: false
license: mit
short_description: Cheaper LLM evaluation and routing, live on real data
tags:
  - evaluation
  - llm
  - bandits
  - routing
  - leaderboard
---

# frugal-eval demo

Methods from the research literature for spending fewer LLM calls on evaluation and routing,
running live on real data: per-question results of 395 Open LLM Leaderboard models, RouterBench
(11 LLMs with real prices) and Chatbot Arena human votes. No API key needed.

- **Find the best model** with as few calls as possible, with a certified winner
- **Estimate scores** of every model from a few questions
- **Route requests** to the cheapest model that is good enough, learning online
- **Pairwise votes**: find the best model from "A or B?" comparisons

Code, benchmarks and papers: https://github.com/raja-sunkara/frugal-eval
