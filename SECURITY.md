# Security Policy

## Supported versions

Only the latest release and `main` receive security fixes while the project is pre-1.0.

## Reporting a vulnerability

Do **not** open a public issue. Use GitHub's
[private vulnerability reporting](https://docs.github.com/en/code-security/security-advisories/guidance-on-reporting-and-writing-information-about-vulnerabilities/privately-reporting-a-security-vulnerability)
on this repository. We aim to acknowledge reports within 7 days.

## Scope notes

- `src/vinfer/server/` is a development server. Do not expose it to the public internet without
  authentication and rate limiting in front of it.
- Weight loading uses `safetensors` only. Pickle-based checkpoints (`.bin`, `.pt`) are not loaded.
