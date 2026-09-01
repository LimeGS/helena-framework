# Staging CI Runner Affinity Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Allow a staging pipeline to build, test, deploy, and smoke entirely on `gpu-1` while retaining `swisspost-1` affinity for non-staging pipelines.

**Architecture:** GitLab workflow rules set one pipeline-wide runner variable from the branch. The local CI-image producer, its unit-test consumer, and the panel-image build use that same variable in their tags; deploy and smoke remain explicitly pinned.

**Tech Stack:** GitLab CI YAML, Python, PyYAML, pytest.

## Global Constraints

- Follow `docs/superpowers/specs/2026-08-02-staging-ci-runner-affinity-design.md` exactly.
- Do not duplicate jobs or manually build/deploy outside GitLab.
- `staging` must resolve the shared CI runner tag to `gpu-1`.
- Non-staging pipelines must default to `swisspost-1`.
- `build the ci image`, `unit tests`, and `build the panel image` must use the same shared tag variable.
- Deploy, smoke, heavy, queue, mission, database, and scientific behavior must remain unchanged.
- TDD is mandatory: the focused test must fail for the old fixed tags before `.gitlab-ci.yml` changes.

---

### Task 1: Branch-aware CI runner affinity

**Files:**
- Modify: `tests/test_the_deployed_frontend_is_the_one_in_the_repository.py`
- Modify: `.gitlab-ci.yml`

**Interfaces:**
- Consumes: `CI_COMMIT_BRANCH` at pipeline creation.
- Produces: one shared runner-tag variable used by the prepare, unit, and panel-build jobs.

- [ ] **Step 1: Write the failing regression test**

Replace fixed-tag expectations for the three Docker-daemon-coupled jobs with assertions that:

```python
parsed["variables"]["HELENA_CI_RUNNER"] == "swisspost-1"
parsed["workflow"]["rules"][0] == {
    "if": '$CI_COMMIT_BRANCH == "staging"',
    "variables": {"HELENA_CI_RUNNER": "gpu-1"},
}
parsed["workflow"]["rules"][-1] == {"when": "always"}
```

and that each of `build the ci image`, `unit tests`, and `build the panel image` has exactly `tags: ["$HELENA_CI_RUNNER"]`. Keep the existing fixed-tag assertions for deploy, smoke, and heavy jobs. Also update the local-CI-image test to assert producer and consumer have the same variable tag.

- [ ] **Step 2: Verify RED**

Run:

```bash
python3 -m pytest -q tests/test_the_deployed_frontend_is_the_one_in_the_repository.py
```

Expected: failure because the workflow mapping and shared runner tag do not yet exist.

- [ ] **Step 3: Implement the minimal CI mapping**

Add a default global variable:

```yaml
variables:
  HELENA_CI_RUNNER: "swisspost-1"
```

Add before `stages`:

```yaml
workflow:
  rules:
    - if: $CI_COMMIT_BRANCH == "staging"
      variables:
        HELENA_CI_RUNNER: "gpu-1"
    - when: always
```

Change only the three specified jobs to:

```yaml
tags: ["$HELENA_CI_RUNNER"]
```

Update the explanatory CI comments so they describe the branch-aware placement.

- [ ] **Step 4: Verify GREEN and regression scope**

Run:

```bash
python3 -m pytest -q tests/test_the_deployed_frontend_is_the_one_in_the_repository.py
python3 -m pytest -q tests/test_a_deploy_updates_every_service.py tests/test_the_platform_backs_itself_up.py tests/test_qc_scales_per_gpu.py
git diff --check
```

Expected: all tests pass and `git diff --check` exits zero.

- [ ] **Step 5: Commit**

```bash
git add .gitlab-ci.yml tests/test_the_deployed_frontend_is_the_one_in_the_repository.py
git commit -m "fix(ci): run staging prerequisites on gpu-1"
```

- [ ] **Step 6: Publish and verify the exact staging SHA**

Push the feature branch and fast-forward `staging`. Require the exact SHA to pass `build the ci image`, `unit tests`, `frontend`, `build the panel image`, `deploy to gpu-1`, and `smoke on gpu-1`; then resume QC diagnostics Task 4.
