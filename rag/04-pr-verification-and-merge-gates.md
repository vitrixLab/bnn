# 04 — PR #15/#16 verification and merge gates

## Reported state (REPORTED, not rechecked here)
CLINE reported #15 and #16 OPEN, not draft, MERGEABLE, with 16/16 checks passing. It reported rulesets endpoint `[]` and classic branch-protection required-status-check endpoint HTTP 403: “Resource not accessible by integration”.

## Correct interpretation
- Passing status rollup does not establish all applicable repository rules.
- 403 is a permission/auth limitation; required-check configuration is UNKNOWN.
- 404 from classic protection does not prove no rulesets exist; inspect rulesets separately.
- An empty rulesets response is only a reported response unless the call and permissions are independently validated.
- Speculation that missing admin scope explains 403 is not verification.

## Required gate from user's historical CLINE directive
1. Inspect state/draft/mergeability/base/review/checks:
   `gh pr view <n> --repo vitrixLab/bnn --json state,isDraft,mergeable,baseRefName,reviewDecision,statusCheckRollup`
2. Check classic protection for actual base branch:
   `gh api repos/vitrixLab/bnn/branches/<base-branch>/protection/required_status_checks`
3. Check rulesets:
   `gh api repos/vitrixLab/bnn/rulesets`
4. Stop if required information is blocked or UNKNOWN.
5. Merge only if every applicable gate is positively verified and explicit owner authorization exists. No bypass shortcuts.

## Outcome
No merge of #15/#16 is verified in this record. Safe state: HOLD / NOT AUTHORIZED until protection/rulesets are resolved and owner explicitly authorizes merge.