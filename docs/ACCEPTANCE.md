# Clean-install acceptance

This protocol verifies one exact Dithyramba checkout in a fresh Linux or macOS
environment. Run it before treating a commit as a release candidate. The
script records the host and architecture but does not make a laptop model part
of the product contract. It requires POSIX paths, Python `3.11.12`, and
`uv 0.6.14`.

## 1. Prepare a clean environment and checkout

You need:

- Linux, or macOS 13 or newer;
- Git, `curl`, and a network connection for the first dependency downloads;
- `uv 0.6.14`.

Install the pinned `uv` release with its
[versioned official installer](https://docs.astral.sh/uv/getting-started/installation/),
then confirm the host:

```bash
curl -LsSf https://astral.sh/uv/0.6.14/install.sh | sh
uv --version
uname -m
uname -s
```

The host values are receipt metadata, not a hardware requirement. A separate
Python installation is unnecessary: the acceptance script downloads and
isolates its pinned Python `3.11.12` runtime.

Clone the repository and identify the exact revision under test:

```bash
git clone https://github.com/esannikov/dithyramba.git
cd dithyramba
git rev-parse HEAD
git status --short
```

Check out the intended commit or tag before continuing. `git status --short`
must print nothing; retain the commit hash with the acceptance result.

## 2. Rehearse with quick mode

```bash
./scripts/acceptance.sh --quick
```

Quick mode creates an isolated temporary root and then:

1. audits the source tree and verifies the frozen lock;
2. installs pinned Python `3.11.12`;
3. builds an sdist, builds the wheel from that sdist, and inspects both;
4. installs only the wheel into a base-runtime environment;
5. verifies package origin, SQLite FTS5, migrations, and schema fingerprint;
6. runs the public demo and the deterministic 1,000-fragment replay;
7. probes the packaged loopback server.

Success ends with:

```text
PASS: quick artifact/runtime route completed
NOTE: Chromium, PDF subprocess, and full aggregate remain for full mode
```

Quick mode is a rehearsal. It is not the final clean-machine result.

## 3. Run the full gate

From the same clean checkout:

```bash
./scripts/acceptance.sh
```

Full mode repeats the artifact and runtime checks, then creates a second clean
QA environment, installs Chromium, runs the required PDF subprocess test and
browser suite without skips, and executes the complete `uv run check`
aggregate.

Success ends with:

```text
PASS: full clean-install source/sdist/wheel/runtime/browser/PDF acceptance completed
```

Record the commit hash, host details, command, exit status, and complete output.
A pass from a modified checkout or a different source revision is not
equivalent to this gate.

## 4. Preserve logs and diagnose failures

The script deletes its isolated scratch directory by default. For a diagnostic
run, retain it and capture the console transcript outside the repository:

```bash
mkdir -p "$HOME/dithyramba-acceptance-logs"
set -o pipefail
DITHYRAMBA_KEEP_ACCEPTANCE=1 ./scripts/acceptance.sh --quick 2>&1 \
  | tee "$HOME/dithyramba-acceptance-logs/quick.log"
```

Omit `--quick` and use a different log name for the full run. The final output
prints the retained scratch path. Depending on how far execution reached, it
contains distribution hashes, migration and health results, demo output,
server diagnostics, and full-mode PDF and browser logs.

Use the first failing check as the diagnostic boundary. Common failures are an
unsupported host, incorrect `uv` version, lock drift, a release-audit finding,
a package content mismatch, an isolated import failure, or a skipped
browser/PDF test.
Fix the underlying checkout or host and rerun from a clean state. Do not publish
the retained scratch directory wholesale: it contains temporary runtime data
and may contain the short-lived loopback bearer token in `serve.stderr`.

## What this gate proves

Neither mode installs Dithyramba's optional semantic extra, provisions a model,
or requires a GPU or API key. The core gate validates the packaged local FTS5
route and its engineering contracts. Full mode may download Chromium, but it
does not test semantic-model quality.

The full gate uses the development aggregate's combined coverage threshold of
`>=95.00%`. Statement and branch coverage are reported separately so a strong
statement score cannot hide weak decision-path coverage.

Passing clean-install acceptance does not establish source truth, retrieval
superiority, semantic accuracy, human usefulness, or production security.
Those require separate evidence and review.
