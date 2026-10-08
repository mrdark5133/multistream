# COMMITS.md: Individual Contributor Commit Guide

Repo: https://github.com/mrdark5133/multistream
Project: HNX26EPS05, Multi-Stream Video Intelligence with Conversational Query

This file defines who authors which commits, how to make them, and how to check them. The build agent (Antigravity) follows it. Copy it to the repo root.

---

## 2. Contributors

| Member | GitHub username | Commit name | Commit email |
|---|---|---|---|
| M1 | irfanbasha11012007-max | `irfanbasha11012007-max` | `irfanbasha11012007@gmail.com` |
| M2 | harivarman-007 | `harivarman-007` | `harivarman124@gmail.com` |
| M3 | mrdark5133 | `mrdark5133` | `mrdark5133@gmail.com` |
| M4 | haygen04 | `haygen04` | `hays2498@gmail.com` |

Notes:
- Commit emails are visible to anyone who can see the repo. A member who prefers privacy can use their GitHub noreply address (shown in GitHub, Settings, Emails) and tell the team to update this table.
- GitHub links a commit to an account only when the commit email is a verified email on that account. If a commit shows no avatar or no profile link, check the email.

---

## 3. Ownership by area (default plan, change if the real split differs)

| Owner | Area | Phases | Paths (from the suggested layout) |
|---|---|---|---|
| harivarman-007 | Environment, model feasibility, ingest pipeline | 0, 1 | `scripts/check_env.py`, `requirements.txt`, `config/`, `src/ingest/`, `src/index/`, `scripts/ingest.py`, `DECISIONS.md` (Phase 0/1 entries), `PLAN.md`, `ARCHITECTURE.md`, `RULES.md`, `TASKS.md`, `.gitignore`, `.env.example` |
| irfanbasha11012007-max | Query engine | 2 | `src/query/`, `src/llm/`, `scripts/query.py`, tests for parsing, time and search |
| mrdark5133 | Clarify-once memory, API, chat UI | 3, 4 | `src/memory/`, `src/api/`, `src/ui/`, tests for aliases and endpoints |
| haygen04 | Evaluation, labeled queries, docs, write-up | 5, docs | `EVAL.md`, `scripts/eval.py`, `eval/`, `README.md`, write-up draft |

Later phases:
- Phase 6 (accuracy fixes): the owner of the code being changed commits it.
- Phase 7 (live mode, re-ID, merge or privacy notes): live mode and watcher go to harivarman-007; re-ID goes to whoever builds it; privacy notes go to haygen04.
- A phase report (`REPORTS/PHASE_N_REPORT.md`) is committed by the member who ran and reviewed that phase.

If one phase touches another owner's files, make **a separate commit for those files** under that owner.

---

## 4. Rules for every commit

1. One logical change per commit. Message format: `phaseN: short imperative summary` (for example `phase1: add start-time fallback chain`).
2. Never commit secrets or large files: `.env`, `.venv/`, `footage/`, `live_footage/`, `index_*/`, model weights, snapshots, clips. Confirm they are ignored.
3. Never use `--no-verify`, never rewrite shared history, never force push.
4. Never use `git config --global` for this project. Set the author on each commit with `--author`.
5. The build agent **commits locally only**. A human runs `git push`. The agent never handles GitHub credentials or tokens.
6. Do not claim a commit contains tested work unless the phase report shows the test output.

---

## 5. Commands (Windows PowerShell)

Check the remote once:

```powershell
git remote -v
git remote add origin https://github.com/mrdark5133/multistream.git   # only if no origin exists
git branch -M main
```

Review before committing:

```powershell
git status
git diff --stat
```

Commit one owner's files (example for M2, replace paths and message):

```powershell
git add scripts/check_env.py requirements.txt config/ PLAN.md ARCHITECTURE.md RULES.md TASKS.md DECISIONS.md .gitignore .env.example
git commit --author="harivarman-007 <harivarman124@gmail.com>" -m "phase0: environment setup, docs and model feasibility check"
```

Author templates for the other members:

```powershell
git commit --author="irfanbasha11012007-max <irfanbasha11012007@gmail.com>" -m "phase2: <summary>"
git commit --author="mrdark5133 <mrdark5133@gmail.com>" -m "phase3: <summary>"
git commit --author="haygen04 <hays2498@gmail.com>" -m "phase5: <summary>"
```

Note: `--author` sets the author. The **committer** is whoever runs the command, based on the local git config. That is normal. If you want both fields to match, each member should commit from their own machine, or set the committer for the one command with environment variables:

```powershell
$env:GIT_COMMITTER_NAME="harivarman-007"; $env:GIT_COMMITTER_EMAIL="harivarman124@gmail.com"
git commit --author="harivarman-007 <harivarman124@gmail.com>" -m "phase1: <summary>"
Remove-Item Env:GIT_COMMITTER_NAME, Env:GIT_COMMITTER_EMAIL
```

Only do this when the named member really approved the commit.

---

## 6. Verify before pushing

```powershell
git log --format="%h | author: %an <%ae> | committer: %cn <%ce> | %s" -n 20
git status
```

Check:
- every commit has the intended author,
- no `.env`, `.venv`, footage, index or model files appear (`git show --stat HEAD` for the last commit),
- the messages match the work.

If an author is wrong and the commit has **not been pushed yet**, fix the latest commit only:

```powershell
git commit --amend --author="Name <email>" --no-edit
```

For older unpushed commits, ask before rewriting history.

---

## 7. Push (human only)

```powershell
git push -u origin main
```

Requirements: the person pushing needs write access to `mrdark5133/multistream` (owner or collaborator) and a working GitHub login on that machine (browser sign-in through Git Credential Manager, or a personal access token entered by that person, never shared in chat or committed). After pushing, open the repo's commit list on GitHub and confirm that each commit shows the right contributor.

---

## 8. Commit log to include in each phase report

Paste the raw output of these commands into the phase report:

```powershell
git remote -v
git log --format="%h | %an <%ae> | %s" -n 20
```

The reviewer checks that the authors match section 3 and that no secrets or large files were committed.

---

## 9. Changing this plan

If ownership changes (for example a member takes over a phase), edit sections 2 and 3 first and commit that change with the message `docs: update commit ownership`. Do not retroactively change authors of pushed commits.
