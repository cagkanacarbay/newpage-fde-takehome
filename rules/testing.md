# Testing

Testing conventions and end-to-end (e2e) policy for this repo. `AGENTS.md` only points
here — this file is where the detail lives.

## Conventions

- Tests live in a top-level `tests/` mirroring the package: `src/<pkg>/sub/foo.py` →
  `tests/sub/test_foo.py`, named `test_<module>.py`. No `__init__.py`.
- Classify scope with **markers** (`@pytest.mark.e2e`, `integration`, `slow`), not
  folders.
- Prefer real inputs. Mock only narrow failure paths and hard-to-trigger edges — never
  the code under test's own collaborators.
- Use pytest-native `assert`, not `unittest` methods.

## End-to-end tests

Prove a change by exercising the product the way a real user hits it, not by calling
internals. Tag these `@pytest.mark.e2e`; they run separately from the fast gate
(`uv run pytest -m e2e`) and are excluded from it, because they drive the real app and
are slow.

Prefer an E2E test for behavior that crosses a process or I/O boundary: HTTP routes,
config loading, file upload and ingestion, streaming responses, stdout/stderr.

**For a bug fix, reproduce the bug end to end first**, as close to how a real user would
hit it as possible, so the fix addresses the real problem. Prove the one user-visible
behavior the change touches, capture evidence, and keep the pass narrow unless a broad
sweep is asked for. Run against isolated, throwaway state — never real data or secrets.

## Building the verification mechanism

As the stack settles, develop the verification mechanism by actually verifying the full
setup once: how to start the app, how an agent drives it, how to capture evidence.
Record that in `rules/verification.md`.
