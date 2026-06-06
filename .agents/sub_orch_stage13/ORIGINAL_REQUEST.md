# Original User Request

## Initial Request — 2026-06-05T19:22:51-06:00

Resume work as the Stage 13 Sub-Orchestrator.
Your working directory is /home/b/microgpt/.agents/sub_orch_stage13.
Read SCOPE.md in your working directory and reference specifications in /home/b/microgpt/phase_3_specs.md.
Your task is to:
1. Decompose the implementation of Stage 13 (Context-Sensitive Data Generation) in src/data/context_sensitive.py.
2. Coordinate worker, reviewer, challenger, and auditor subagents to:
   - Implement the Copy Task (ww) and Multiple Counting (a^n b^n c^n) sequence generators.
   - Ensure they are formulated as next-token prediction tasks.
   - Run tests to verify correctness of data shapes and vocabulary mappings.
3. Report back when complete.
As a sub-orchestrator, follow the orchestrator procedure: decompose, plan, delegate to worker/reviewer/challenger/auditor subagents. Do NOT write code directly.
Your parent is 3e03d858-a59f-4e4e-966c-3a9ee099fb56. Report back via send_message.
