# Live Long R&D Research Assistant

A chat with your docs assistant based on a hypothetical customer "Live Long R&D" whose researchers work on longevity. They need an assistant that is focused on their domain, can help them parse through research in their field, find it reliably, accurately, and provide them with the knowledge in those research papers to power their own work. 

## Installation

<!-- Maintained by the agent: keep accurate and updated with every change that affects setup -->

Prerequisites: `uv` (Python 3.12), Node 22 with `pnpm` 10, and an OpenAI API
key. The same key serves embedding, live answer generation, and claim verification.

1. `uv sync` - install the Python stack.
2. Put `OPENAI_API_KEY=...` in `.env` (gitignored). The index build and live
   application need it. The stub modes below need no key.
3. Build the vector index once:

   ```bash
   uv run python -m live_long_rnd.ingest data/corpus/longevity
   ```

   This parses all 27 PDFs with Docling, embeds with OpenAI
   `text-embedding-3-large`, and writes a self-contained LanceDB index to
   `data/index/` (gitignored; fully reproducible with this command).
4. Build one application image with the completed index:

   ```bash
   docker build \
     --build-context index=data/index \
     --tag live-long-rnd .
   ```

   The image contains the Next.js export and all 696 LanceDB chunks.
   It does not contain `.env`, the PDF corpus, Docling, Torch, or the OpenAI key.
5. Start the complete application on port 8000:

   ```bash
   mkdir -p data/state
   chgrp "$(id -g)" data/state
   chmod 0770 data/state
   docker run --rm --env-file .env --publish 8000:8000 \
     --group-add "$(id -g)" \
     --volume "$(pwd)/data/state:/app/state" \
     live-long-rnd
   ```

   Open http://localhost:8000.
   The runtime key serves query embeddings, answer generation, and claim verification.
   The bind mount keeps conversations in `data/state/conversations.db` after the
   container stops.

For local development, run the API and UI as separate processes.
The API defaults to deterministic stubs when the live adapter variables are absent.

Tests: `uv run pytest` (full suite, includes end-to-end tests that parse a
real PDF and build a real index) or `uv run pytest -m "not e2e"` for the fast
unit suite.
Set `RUN_DOCKER_E2E=1` to include the one-image container smoke test.
Lint and types: `uv run ruff check`, `uv run mypy`.

The repository includes the 27-PDF research corpus in
`data/corpus/longevity/`.
Use `uv run scripts/fetch_corpus.py` only to restore or refresh the corpus from
its public sources.

## Architecture overview

<!-- Cha writes this section himself. Leave empty. He may ask the agent for specific parts. -->

## Productionizing, scaling, and hyperscaler deployment

<!-- Cha writes this section himself. Leave empty. He may ask the agent for specific parts. -->

The entire system is self contained within the docker app. Of course this is only for this demo to make it easy to run it. 

For productionizing it, we would need to update a few things:
1. The database is currently a sqlite. It would need to be migrated to a Postgres and hosted on a managed database provider. 
2. Parsing and embedding will need to be queued and moved in the background jobs. This would allow us to add retries, idempotency, dedpulication and index versioning, which are critical to maintaining this system long term.
3. We need a file storage system such as Amazon S3. Currently files are simply a part of the Docker image. We would serve these with signed URLs.
4. The current LanceDB would be replaced with a managed store. LanceDB Cloud or Qdrant Cloud, or an a VPC alternative if the customer requires data to be in their own cloud.
5. Authentication and authroization would need to be added through the identity provider of the customer. Currently no auth implemented.
6. The current eval set is relatively simple. I would develop with the customer a gold retrieval set and run it before each release to prevent regressions. Another improvement here would be to build in user feedback so users can provide guidance on how the system is failing, so we can use that info to improve LLM and retrieval performance.


## Key technical decisions and why

### Assignment
Chose to build a longevity research paper corpus, as it might be an actual customer use case, building assinstants that help researchers do their work in fields such as longevity.

### LLM Choice
GPT 5.6 Luna handles all parts of the workflow. It is relatively cheap and very performant. It handles:
- query planning, answer generation, claim verification

Main generation uses high reasoning. High reasoning is generally where the benefits of reasoning are best vs the cost of more reasoning. Although in a production environment I would test this as our specific use case might benefit from extensive reasoning effort to handle highly specific scientific information and data.

Verifier uses low reasoning simply checking chunks vs the result.

### RAG/LLM approach and decisions

<!-- Cha writes this section himself. Leave empty. He may ask the agent for specific parts.
     Covers: LLM, embedding model, vector database, orchestration framework,
     prompt & context management, guardrails, quality, observability. -->

There are many RAG options that provide end to end RAG capabilities such as RAG Anything. I chose not to use such read frameworks to show my thinking and approach to each of the parts of a RAG pipeline separately. 

#### Framework

Chose llamaindex as the framework for our RAG pipeline because the selected docs are scientific papers, PDFs, with lots of tables, equations, graphs, and llamaindex provides a good, simple way to parse and feed them into the pipeline. Provides the full set so we can run tests on our data and pick the best method to chunk, store, retrieve the data, plus build evals on the same pipeline. 

#### Parsing and Chunking
Considered multiple parsers to convert the 27 target PDFs into structured markdown. This is where a lot of RAG pipelines fail. Converting PDFs into markdown will have issues so a great parser is necessary. 

Researched and tested multiple solutions for parsing. Chose Docling as it:
1. is open source and usable by an enterprise client without any fees
2. tested well for our corpus:
    a. tables don't break
    b. retrieval collision with repeated headings in the same paper
    c. provides document hierarchy which can be used for citations
3. works well with llamaindex
4. works locally on CPU so I can run it without issues for my demo

My chunking strategy is the widely accepted strategy the broader RAG engineering community is converging upon for production systems:
1. Structural chunking to preserve meaning across the structure of the document, since fixed size chunking messes up the chunks and removes all meaning. 
2. Metadata enrichment of chunks so we can both filter out retrieval chunks by identifiers, and receive useful information in the chunks themselves.
3. Contextual chunks: We add context to each chunk so the LLM can understand the chunk as part of a larger document and place it within that context as it analyzes. At this stage the added context is deterministic. We add paper, page number, section, etc. A natural improvement would be to run a model to write contextual chunks for each chunk that place the context in prose. 

#### Vector DB
For our vector database I chose LanceDB as it allowed me to use locally and ship it as part of our Docker image. 

#### Embedding
Using text-embedding-3-large from GPT since it is available through the same API key I have for GPT. For a production system I would use a stronger model, either host it on the customer systems, or through a trusted provided.

Qwen3-Embedding-8B would be a good upgrade in production.

#### Retrieval
The corpus, while small, is very difficult corpus with a lot of highly scientific papers, close semantic meanings, a lot of specific keywords that are only available through sparse vectors. 

The general approach of course is to apply BM25 + Dense Vector search + RRF but the way I have implemented this is through a query planning approach where each user query is turned into semantic and sparse vector search items. So user queries get turned into proper search terms even if the user query is not very clear. See below this structure.

```json
  "search_intents": [
    {
      "dense_query": "What doses and adverse events were reported in the Hickson senolytic trial?",
      "sparse_query": "Hickson senolytic dose adverse events",
      "filters": {}
    }
```

Another benefit is that these items can be stored, tracked on performance if retrieval succeeeds or not, cached later on in a production system to improve. This is not implemented but it's a natural improvement.

I've also added MiniLM cross-encoder reranker as a final step to rerank each candidate against the question itself. This was added to improve the retrieval quality after my evals returned weaker results.

#### Context Management
This has been kept simple as other areas have taken more focus for the initial system. 

A simple approach, I chose a rolling 100k input token window for the chat. Longer chats will simply not include the previous parts. 

An extensive discussion of how this could be productionized, extended, and made more functional is provided in the section What I'd do with more time.

#### Guardrails 
The main guardrail that I implemented is a verifier check with a GPT 5.6 Luna model, that reads the main models response, checks the retrieved chunks, identifies exactly what the main model took from the chunk and returns that. This is what allows us to return highlighted citations. So each and every one of the returned items is reliably from a source material. 

#### Quality
I created an eval set from the corpus that runs 24 questions against the corpus set against the LanceDB index. 
1. 6 questions test both sides of a scientific disagreement are captured
2. 6 test similar entity names and distractions
3. 12 tests require a known paper in top retrieved results. 

Quality tests run on CI so any change touching retrieval must not regress the reesults.

Also running on CI:
  - Python formatting, Ruff, strict mypy, unit tests, E2E tests, and
    the Docker smoke test.
  - A path-scoped 24-item retrieval-quality gate.
  - Web tests, ESLint, and the production Next.js build.

### Observability
Originally I wished to set up Arize Phoenix with the repository to build in observabillity right into the system. Time did not allow it. 

## Engineering standards followed (and skipped)

### Ruff linting and checks
Enforcing code quality through automated ruff lint and checks that fire through hooks. 

Each coding agent edit enforces the hooks to fire and check for violations. Violations get caught immediately so code quality from this perspective is deterministic.

The rule set I applied are my own I use for any new project. I might go in and do some changes based on the specific work at hand but I have my defaults set for this one since its a simple demo repo. 

One key thing I have enforced is function length. It ensures the coding agent must abide to smaller function definitions which generall tends towards cleaner, easy to understand code.

### Automated Tests 
Tests are cheap in the age of LLMs, so I take advantage of that by running them, all the time. Unit tests take miliseconds to run so they run after each code change, forcing the coding agents to always consider the health of the codebase. Any caused error is immediately noticed, as long as a test covered that case.

### Test Driven development
Coding agents write all the code. The way they do this is a Red -> Green, Red -> Green, ... TDD methodology so they write one failing test, see it fail, write the code that passes the test, and then write another failing test. This forces the LLM to write better more useful tests. It doesn't mean all tests it writes are useful, but more so than not so the overall impact is gradually having a god set of tests that cover important aspects of the codebase.

The thing about this is, with LLMs, tests are free. Having LLMs write these tests all the time increases the chances of catching issues early on. 

### Write a test when there is an issue found
Any issue found, a bug, an unconsidered problem, gets its own test through the previously described TDD methodology. This makes sure real world bugs, problems encountered etc. then feed back into the system as improvements of the codebase. The tests run all the time, so these issues never come back.

### Test Pruning
Adding a bunch of tests and running them on auto makes the codebase brittle long term. To make tests pass the codebase will be a certain way and will evolve with a lot of limitations.

Every so often, depending on development velocity, I would run a specific analysis of the tests, and prune them, then refactor the code, making sure code quality stays high and codebase can evolve in this manner.

### git worktrees with Treehouse
Treehouse is a tool that allows AI agents to manage git worktrees so multiple branches could be owned by AI agents to develop in parallel without colliding. It allows isolation of work and a faster reliable pipeline. There isn't much room to showcase what this enables with this repo, but I have it on for all my projects as it adds a lot to my agentic workflow.

### no-mistakes as the PR gate
This system takes work done by the LLM and runs it through a gauntlet of review, testing, and refactoring so my job at the end becomes reviewing a PR that's been analyzed. Not all PR's are created equal, and this allows for me to spend more time on high risk PRs that touch important systems, while spending less time on low risk PRs. I have it on all my repos where code quality is important. 

### Automated UI Tests
Whenever UI is updated, it is picked up through hooks, then a Codex agent goes in takes screenshots, works through the UI and visually verifies it. This fixed many simile UI issues out of the gate. I have this on for UI projects, but it can be token hog, so I'm looking at ways to control when and how its used more effectively. 

## How AI tools were used in development

Coding agents are my primary work surface.

First we scope the work, discuss options, and architecture on HTML docs. See the docs for some artifacts.

See docs/index.html. It includes the technical choices I evaluated, the research I made to make the choices, and decisions I made. I use the HTML surface as both an alignment surface with AI agents, and to communicate to teammates/customers. 

After work is scoped, the agents handle implementation through the TDD system, and other agents run to verify and validate. My work is at the beginning when scoping the work, and at the end when validating and verifying.

Agents run automatically for verification, review, and after code changes break tests, as described above. 
The coding agents' coding capabilities are constrained by the myriad of tests, lints, rules, deterministic checks that force it to work in a certain way. 

I am always looking at ways of doing this meta-work of constraining how the agent works. The agents are great at spewing out code. It's on me as the engineer to build the harness around it so that it can create code as fast as possible, but I trust that the code is up to my quality standards. Those standards and the speed to code must always be increasing, as both models and our ways to harness engineer improve. 

## What I'd do differently with more time

This projects was a demo that I based on a research paper stack I selected. What I'd do differently depends mostly on what the actual requirements of a work would be. I can define the problem I was working on more specifically, and place it in a hyphotetical customer environment and answer.

Customer: Big pharma research company working on R&D across the longevity and related fields. Researchers looking to utilize AI to improve their R&D process. More specifically the ask from them is to have a dedicated assistant who has all the work from the emerging field, and can assist in identifying relevance to the researcher's own work in others' papers.

Given this scenario I would:
1. Build a pipeline that feeds in fresh papers into the system. The customer might want control over what papers get ingested, so this might be a simple UI that brings them relevant papers for them to choose from, OR it might be a UI where they give the system the papers. 
2. Ship MVP. Observe customer behavior if possible. use customer behavior to create eval sets. 
3. Add in a way for the customer to flag issues they run into. Ask AI -> AI returns wrong/hallucinated answer -> user flags it -> I fish out the reason, turn it into an eval -> similar issues get flagged by eval.
4. Focus on understanding the customers' use case. It's easy to build a RAG pipeline that simply answers questions. But that's never the real goal of the customer. The customer wants a tool to make their work easier. What does that work entail? Does it cross these research papers as well as their own work? Is their work avaiable to be indexed? Do we need access controls? A boundary between their work and these papers? What is the actual goal of the researcher? What painful, time consuming tasks can our system look to eliminate or alleviate? Questions of this sort need to be understood and turned into solutions. 
5. Authentication has not been considered as part of this work. In a real scenario we would deploy within the authentication paradigm of the customer.
6. Work with the actual corpus, which would likely be thousands of papers, as well as other sorts of documents, to select each part of the pipeline. I'd run tests similar to the ones I've ran at a smaller scope in this repo, understand the corpus' nature, which chunking methodology provides better results in retrieveal and other tests. 
7. Parent-child retrieval. A likely addition I would make to this system. This improves the general reliability of the system. It would require another data store to store the parents and add a level of complexity that is overengineering for the purposes of this simple demo, but a production system would benefit from this.
8. Conversation management is kept very simple. It's a simple session system. User can start a new conversation or continue older one. There is no compaction. No context management within and beyond sessions. This would be an area of improvement in any production system. For long chats only the last 100k tokens are used as input. Easy wins here are compaction, session summaries, memory system that stores user queries, and findings when the user specifically asks, or makes what seems to be important connections. This I would develop after understanding actual user patterns, as memory could easily be a negative rather than a positive.
9. For simplicity sake the corpus raw PDFs are part of the docker image and served directly through there. We use that to directly cite the papers in the chat. In a production system we would need to serve them through a filesystem. 
10. Build in observability. Each and every chat with the LLM, as well as each retrieval event should be observed and tracked. I was looking to build it in with Arize Phoenix this functionality. I would if I had more time.
11. 


## Screenshots

<!-- Screenshots of the running application go here. -->
