---
name: gitcode-security-check
description: Perform a repository security and sensitive information review for GitCode-hosted code. Trigger for security audit, secret scan, credential leakage, dependency risk, injection risk, auth risk, or pre-release security review.
metadata:
  version: 1.0.0
  source: gitcode-cli/skills@6f0d8b0-adapted
---

# gitcode-security-check

Review a repository for sensitive information and security risks.

## Command Entry

Use `gitcode` for cross-platform instructions. Linux/macOS may use `gc`; Windows PowerShell should use `gitcode`.

## Scope

Clarify:

- Repository: `owner/repo`
- Branch or PR
- Paths to include or exclude
- Whether local write actions are allowed

Clone with SSH when code is needed:

```bash
ssh -T git@gitcode.com
gitcode repo clone owner/repo --git-protocol ssh
```

## Scan Checklist

Use `rg` if available; fall back to platform tools only when necessary.

Secrets and credentials:

```bash
rg -n --hidden -S "(token|api[_-]?key|secret|password)\s*[:=]\s*['\"][^'\"]+" .
rg -n --hidden -S "-----BEGIN (RSA |EC |OPENSSH )?PRIVATE KEY-----" .
git ls-files | rg -i "(\.env$|\.pem$|\.key$|credentials\.json$|id_rsa|id_ed25519)"
```

Token-like values:

```bash
rg -n --hidden -S "pypi-[A-Za-z0-9_-]{20,}|ghp_[A-Za-z0-9_]{20,}|gitcode[_-]?[A-Za-z0-9_-]{20,}" .
```

Injection and unsafe execution:

```bash
rg -n -S "exec\.Command|subprocess\.|popen|system\(|eval\(|fmt\.Sprintf.*SELECT|SELECT .* \+" .
```

Sensitive logging:

```bash
rg -n -S "log\..*(token|secret|password)|console\.log.*(token|secret|password)|fmt\.Print.*(token|secret|password)" .
```

Dependency and config clues:

```bash
git ls-files | rg "go\.mod|go\.sum|package-lock\.json|pnpm-lock\.yaml|requirements.*\.txt|poetry\.lock|pom\.xml|build\.gradle"
rg -n -S "replace\s|http://|tlsSkipVerify|InsecureSkipVerify|verify=False|strict-ssl false" .
```

## Report

```markdown
## Security Review

### Scope
- Repository:
- Branch/PR:
- Paths:

### Findings
- [Critical] file:line - issue, impact, fix.
- [High] file:line - issue, impact, fix.

### Sensitive Information
- Confirmed secrets:
- False positives:
- Rotation needed:

### Residual Risk
- ...
```

## Rules

- Do not print full secrets; show prefixes or hashes only.
- If a real secret is found, recommend revocation and rotation.
- Separate confirmed findings from pattern matches.
- Do not modify repository history unless the user explicitly requests a remediation workflow.
- For PR security review, inspect changed files first, then shared auth/config paths.
