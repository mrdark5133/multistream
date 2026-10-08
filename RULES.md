# NON-NEGOTIABLE RULES

## 1.1 Honesty rules (most important)
1. **No fake results.** Never invent, estimate, round up, or "expect" a number, test result, log line, accuracy, latency, VRAM figure, or file size. Every number in a report must come from a command you actually ran, and the raw output must be pasted into the report.
2. **No silent mocking.** If you use synthetic, random, or placeholder data for any test, label it **SYNTHETIC** in the report, and never count it toward accuracy or performance claims.
3. **Say "NOT RUN" or "NOT VERIFIED"** for anything you did not execute. Never write "should work", "works", or "passes" about something you did not run.
4. **Do not create ground truth.** Evaluation labels (`eval/queries.json`) are written by the human team. You may build the harness but must not write labels, and must not tune on the held-out split.
5. **If a test fails, report the failure** with the full error text. Do not delete the test, weaken it, or hide it. A failed phase is reported as FAILED or PARTIAL.
6. **Never claim a hardware or model capability from memory.** Check it (for example by running `nvidia-smi`, loading the model, reading file sizes).
7. If something in this prompt is wrong or impossible (for example a library API differs), say so in the report under "Deviations" instead of working around it silently.

## 1.2 Browser and screenshot rules
1. **Do NOT use the browser tool or browser agent.** Do not auto-open a browser, do not take screenshots, do not record screen captures, and do not run browser-based verification.
2. Verify everything through the terminal: unit tests, `pytest`, `curl`/`httpx` calls to the API, SQLite queries, file listings, logs.
3. If a UI behavior truly needs a human to look at it, write a **MANUAL CHECK** item in the report (exact URL, exact steps, expected result) and leave it for the human. Mark it NOT VERIFIED until the human confirms.

## 1.3 Working rules
1. Follow the phases in order. Do not start a phase early. Do not build features from later phases.
2. Keep changes small and reviewable. Commit at the end of every phase with a clear message, and put the commit hash in the report.
3. Never commit secrets. API keys come from environment variables only. Provide `.env.example`.
4. Never commit large files: models, videos, snapshots, the index. Add them to `.gitignore`.
5. Ask the human a question (and stop) if you need something only they can provide: footage, API key, labeled queries, a decision.
6. Prefer simple code that works over clever code. Add logging for every pipeline stage.
7. Do not delete or overwrite user footage or an existing index. Use new output folders.
