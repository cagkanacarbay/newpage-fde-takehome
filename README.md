# Live Long R&D Research Assistant

A chat with your docs assistant based on a hypothetical customer "Live Long R&D" whose researchers work on longevity. They need an assistant that is focused on their domain, can help them parse through research in their field, find it reliably, accurately, and provide them with the knowledge in those research papers to power their own work. 

## Installation

<!-- Maintained by the agent: keep accurate and updated with every change that affects setup -->

There is no runnable application yet.

The repository includes the 27-PDF research corpus in
`data/corpus/longevity/`.
Use `uv run scripts/fetch_corpus.py` only to restore or refresh the corpus from
its public sources.

## RAG/LLM approach and decisions

<!-- Cha writes this section himself. Leave empty. He may ask the agent for specific parts.
     Covers: LLM, embedding model, vector database, orchestration framework,
     prompt & context management, guardrails, quality, observability. -->

### Framework

Chose llamaindex as the framework for our RAG pipeline because the selected docs are scientific papers, PDFs, with lots of tables, equations, graphs, and llamaindex provides a good, simple way to parse and feed them into the pipeline. Provides the full set so we can run tests on our data and pick the best method to chunk, store, retrieve the data, plus build evals on the same pipeline. 

## Key technical decisions and why

### Assignment
1. Chose to build a longevity research paper corpus, as it might be an actual customer use case, building assinstants that help researchers do their work in fields such as longevity.

### RAG pipeline
1. Choose the pipeline according to the data. 

## Engineering standards followed (and skipped)

### Ruff linting and checks
Enforcing code qualiy through automated ruff lint and checks that fire through hooks. 

Each coding agent edit enforces the hooks to fire and check for violations. Violations get caught immediately so code quality from this perspective is deterministic.

The rule set I applied are my own I use for any new project. I might go in and do some changes based on the specific work at hand but I have my defaults set for this one since its a simple demo repo. 

One key thing I have enforced is function length. It ensures the coding agent must abide to smaller function definitions which generall tends towards cleaner, easy to understand code.

### Automated Tests 
Tests are cheap in the age of LLMs, so I take advantage of that by running them, all the time. Unit tests take miliseconds to run so they run after each code change, forcing the coding agents to always consider the health of the codebase. Any caused error is immediately noticed, as long as a test covered that case.

### Test Driven development
Coding agents write all the code. The way they do this is a Red -> Green, Red -> Green, ... TDD methodology so they write one failing test, see it fail, write the code that passes the test, and then write another failing test. This forces the LLM to write better more useful tests. It doesn't mean all tests it writes are useful, but more so than not so the overall impact is gradually having a god set of tests that cover important aspects of the codebase.

### Write a test when there is an issue found
Any issue found, a bug, an unconsidered problem, gets its own test through the previously described TDD methodology. This makes sure real world bugs, problems encountered etc. then feed back into the system as improvements of the codebase. The tests run all the time, so these issues never come back.

### git worktrees with Treehouse
Treehouse is a tool that allows AI agents to manage git worktrees so multiple branches could be owned by AI agents to develop in parallel without colliding. It allows isolation of work and a faster reliable pipeline. There isn't much room to showcase what this enables with this repo, but I have it on for all my projects as it adds a lot to my agentic workflow.

### no-mistakes as the PR gate
This system takes work done by the LLM and runs it through a gauntlet of review, testing, and refactoring so my job at the end becomes reviewing a PR that's been analyzed. Not all PR's are created equal, and this allows for me to spend more time on high risk PRs that touch important systems, while spending less time on low risk PRs. I have it on all my repos where code quality is important. 

### Automated UI Tests
Whenever UI is updated, it is picked up through hooks, then a Codex agent goes in takes screenshots, works through the UI and visually verifies it. This fixed many simile UI issues out of the gate. I have this on for UI projects, but it can be token hog, so I'm looking at ways to control when and how its used more effectively. 
