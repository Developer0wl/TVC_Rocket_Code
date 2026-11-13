---
name: implement-loop
description: Run the standard implement -> test -> review loop for one small unit of work (a module, a feature, a tuning change). Use for every roadmap step so each step is verified before the next begins.
argument-hint: <one-step task description>
---

Run this loop for the single step described in the arguments. Keep each step small enough to verify.

1. **Spec**: restate the step and its acceptance test in 2-3 lines. Write the failing test first when the behavior is checkable.
2. **Implement**: make the change (main agent, or `controls-engineer` / `rl-engineer` for their areas).
3. **Gate**: delegate to `test-runner`. On failure, fix and re-run. Stop and ask the user after 3 failed iterations on the same failure instead of looping blindly.
4. **Review**: for changes to dynamics, motor, actuator, or atmosphere, delegate to `physics-reviewer`. Fix CRITICAL/HIGH findings and return to step 3.
5. **Record**: update CLAUDE.md only if a convention or command changed. Report what was done and the evidence (tests passing, metrics).
6. Do not commit unless the user asks.
