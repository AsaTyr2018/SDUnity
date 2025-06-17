# Contributing to **SDUnity**

> *Empowering creators by merging Stable Diffusion and open tooling*
> We warmly welcome contributions of **code, documentation, tests, designs, ideas, and enthusiasm!**

---

## Table of Contents

1. [Code of Conduct](#code-of-conduct)
2. [Getting Started](#getting-started)
3. [How to Contribute](#how-to-contribute)
4. [Coding & Style Guidelines](#coding--style-guidelines)
5. [Issue Reporting](#issue-reporting)
6. [Pull Request Process](#pull-request-process)
7. [Testing](#testing)
8. [Branch & Commit Naming](#branch--commit-naming)
9. [Documentation](#documentation)
10. [Community & Support](#community--support)
11. [License](#license)

---

## Code of Conduct

This project follows the [Contributor Covenant](CODE_OF_CONDUCT.md) v2.2.
By participating you agree to uphold these guidelines to ensure a welcoming environment for everyone.

---

## Getting Started

1. **Fork** the repository and clone it locally:

   ```bash
   git clone https://github.com/AsaTyr2018/SDUnity.git
   cd SDUnity
   ```
2. **Pick an issue** (or open a new one) and comment that you are working on it.
3. **Create a virtual environment** (for the Python side) and install dependencies:

   ```bash
   python -m venv venv && source venv/bin/activate
   pip install -r requirements.txt
   ```
4. **Run the main interface** to verify installation:

   ```bash
   ./start.sh
   ```

---

## How to Contribute

| Contribution Type | How to Proceed                                                         |
| ----------------- | ---------------------------------------------------------------------- |
| **Bug Fix**       | Search existing issues → Comment to claim → Fix → PR                   |
| **New Feature**   | Open a feature request → Discuss scope → Draft design → Implement → PR |
| **Documentation** | Fork → Edit docs → PR (no approval required for typo/grammar fixes)    |
| **Translations**  | Create/Update `/docs/i18n/` locale → PR                                |
| **Design/UX**     | Share mock-ups in Discussion → Iterate → PR assets                     |

> **Small improvements are valuable!** Even fixing a typo helps everyone.

---

## Coding & Style Guidelines

### Python (Core & Tools)

* Follow **[PEP 8](https://peps.python.org/pep-0008/)** and use **Black** for formatting (`black -l 120`).
* Type-annotate all public functions. Use `mypy` for static checks.
* Maintain modular, testable design and use `argparse` for CLI tools.

### Shell Scripts

* Prefer POSIX-compliant syntax.
* Always `set -euo pipefail` at the top of scripts.
* Use descriptive variable names and comments for install/update logic.

### Assets & Models

* Commit only **text-based** configuration files. Do **not** commit `.ckpt`, `.safetensors`, `.blend` or `.fbx` binaries directly.
* Store model checkpoints (>25 MB) using Git LFS.

---

## Issue Reporting

1. **Search first** to avoid duplicates.
2. Use the **Bug Report** or **Feature Request** templates.
3. Include **repro steps**, **expected**, **actual behaviour**, and relevant logs (`log.txt`, `stdout` or CLI output).
4. For security vulnerabilities **do not** open a public issue; email `security@sdunity.dev`.

---

## Pull Request Process

1. **Branch off** `main` (for hotfixes) or `develop` (for features).
2. Ensure your code **builds & tests pass** (GitHub Actions will run CI).
3. Keep PRs **focused & small** (≤ 400 LOC diff is ideal).
4. Add a **clear description**: *What*, *Why*, *How tested*.
5. Tag reviewers: `@SDUnity/maintainers`.
6. At least **one approval** + **green CI** is required before merge.
7. Squash-merge unless multiple commits have meaningful history.

---

## Testing

* Unit tests reside in `/tests/` (Python). Shell scripts should include dry-run or simulation flags.
* Write tests for **every bug fix** and for **new public API**.
* Run tests locally:

  ```bash
  # Python
  pytest -q
  # Shell linting (optional)
  shellcheck scripts/*.sh
  ```

---

## Branch & Commit Naming

* **Branches**: `feature/<topic>`, `bugfix/<issue-id>`, `docs/<topic>`.
* **Commits** use **Conventional Commits** (`feat:`, `fix:`, `docs:` …). Examples:

  * `feat(generator): add wildcard support`
  * `fix(install): prevent missing dependency crash`

---

## Documentation

* Source resides in `/docs` (Markdown).
* Update internal documentation strings (docstrings) when changing behavior.
* Add screenshots / gifs to enhance clarity (≤ 5 MB per asset).

---

## Community & Support

| Channel                | Purpose                                        |
| ---------------------- | ---------------------------------------------- |
| **GitHub Discussions** | General questions, ideas, show & tell          |
| **Issues**             | Bug reports & feature requests                 |
| **Discord**            | Real-time chat (see badge in README)           |
| **YouTube**            | Tutorials & showcases (see `AiMusics-Central`) |

We aim for **async-friendly** collaboration; please be patient for review.

---

## License

By contributing, you agree that your work will be licensed under the **GNU General Public License v3.0** unless stated otherwise in the pull request.

See [LICENSE](LICENSE) for full terms.
