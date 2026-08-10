# Live Long R&D assistant

A retrieval-augmented chat assistant for longevity researchers.
Every answer must be grounded in the curated paper corpus with checkable citations.

## Language

**Conversation**:
A persisted sequence of turns between one researcher and the assistant, stored in SQLite.
Starting a new chat starts a new conversation.
_Avoid_: session, thread

**Turn**:
One user message plus the assistant's response to it.
_Avoid_: exchange, round

**History budget**:
The 100k-token cap on how much of a conversation's most recent turns are kept verbatim for the model.
When the budget is exceeded, the oldest turns drop first and the user is told inline.
_Avoid_: context window (that is the model's limit, not ours), sliding window
