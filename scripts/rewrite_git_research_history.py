#!/usr/bin/env python3
"""Rewrite PEMFC Git history into a research-oriented, documented history.

This is intentionally destructive to the selected branch SHAs.  It preserves
file trees, parent topology, authors and dates through git-filter-repo, while
rewriting commit messages only.  A remote backup branch must exist before use.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

BRANCHES = [
    "main",
    "v02-convergence-study",
    "v03-convergence-study",
    "v03-hydrated-membrane",
    "v04-coupled-cathode-membrane",
    "v04-rh-sweep",
    "v05-coupled-membrane-water",
    "v06-anode-water-transport",
]

BACKUP_BRANCH = "backup/pre-history-rewrite-20260928"

MILESTONES = {
    "v0.1": (
        "80a463dbdedd",
        """V0.1 — 3D oxygen-transport cathode baseline

First stable Dedalus cathode model:
- 3D representative GDL/CL domain
- oxygen diffusion and reaction sink
- basic parameter tests
- corrected Dedalus namespaces

Purpose: establish the numerical and software baseline before electrochemistry.
""",
    ),
    "v0.2": (
        "5493b8f83e53",
        """V0.2 — Coupled cathode electrochemistry validated

Adds:
- oxygen transport coupled to ORR kinetics
- electronic and protonic potentials
- Butler-Volmer response
- Ballard-derived operating references
- validation, Ruff, mypy and spatial convergence workflow

The V0.2 convergence study is the validated electrochemical baseline.
""",
    ),
    "v0.3": (
        "7bc49fdf9201",
        """V0.3 — Hydrated membrane model validated

Adds and validates the standalone 1D membrane physics:
- Springer water-content relation
- hydration-dependent proton conductivity
- electro-osmotic drag
- membrane transport parameters
- membrane convergence diagnostics

This isolates membrane constitutive physics before cathode coupling.
""",
    ),
    "v0.4": (
        "7c2977301a89",
        """V0.4 — Hydrated cathode and RH/voltage study

Couples cathode RH to ionomer hydration and proton conductivity.
Validated through:
- corrected protonic charge-conservation sign
- spatial convergence
- RH/voltage sweep
- polarization and conductivity sensitivity analysis

This is the final cathode-hydration baseline before explicit membrane ASR feedback.
""",
    ),
    "v0.5": (
        "883e4ecb50b0",
        """V0.5 — Coupled cathode/membrane water model validated

Partitioned fixed-point coupling:
3D cathode -> current density -> 1D membrane water profile -> membrane ASR
-> proton-potential boundary -> cathode.

Validated through:
- zero-anode-water-flux dead-end approximation
- RH/voltage sweep
- representative spatial-convergence study
- final pytest, Ruff and mypy gate

Reference study grid: 16x16x48.
Validation grid: 32x32x96.
""",
    ),
}

RESULT_SUBJECTS = {
    "27c5996f82e3": "Record V0.2 electrochemistry validation baseline",
    "7baaa6b76276": "Record final V0.2 convergence validation",
    "3b27e6568cd1": "Record V0.3 membrane convergence results",
    "6374e4641ef2": "Record initial V0.4 hydrated-cathode validation",
    "ac13c7a95480": "Record corrected V0.4 proton-sign validation",
    "15f71ef883f7": "Record V0.2/V0.4 quick pipeline validation",
    "0738785f177e": "Record V0.4 quick comparison results",
    "aa6aa4e9d101": "Record corrected V0.4 reference comparison",
    "2978e9ed714b": "Record final V0.4 spatial validation",
    "a38c83596d20": "Record V0.4 RH-voltage sweep results",
    "685d75cb2f70": "Record V0.4 RH sensitivity analysis results",
    "7c2977301a89": "Record final V0.4 RH study and figures",
    "7d870d3c9d9f": "Record V0.5 zero-flux coupling preflight",
    "c4965b382a19": "Record V0.5 nominal coupled solution",
    "08a5986f2a00": "Record V0.5 RH-voltage sweep results",
    "b0e463ec0288": "Record V0.5 quick spatial-convergence results",
    "e3d283e8be88": "Record V0.5 full spatial-convergence results",
    "cb49157df1ba": "Record final V0.5 validation dataset",
    "d40e3a95a075": "Record V0.6 nominal finite-transfer result",
    "4a9c4abe52e0": "Record V0.6 zero-transfer continuity result",
}

SPECIAL_SUBJECTS = {
    "12bb6201c7f6": "Fix Ruff issues in V0.6 coupling code",
    "9438e58526a9": "Document PEMFC research and development history",
}


def run(*args: str, capture: bool = False) -> str:
    proc = subprocess.run(
        args,
        check=True,
        text=True,
        stdout=subprocess.PIPE if capture else None,
    )
    return proc.stdout.strip() if capture else ""


def require_clean_worktree() -> None:
    if run("git", "status", "--porcelain", capture=True):
        raise SystemExit("Refusing history rewrite: working tree is not clean.")


def ensure_filter_repo() -> None:
    try:
        run("git", "filter-repo", "--version", capture=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        raise SystemExit(
            "git-filter-repo is required. On Debian: sudo apt install git-filter-repo"
        ) from None


def remote_branch_sha(branch: str) -> str:
    out = run("git", "ls-remote", "--heads", "origin", branch, capture=True)
    if not out:
        raise SystemExit(f"Remote branch origin/{branch} not found")
    return out.split()[0]


def ensure_local_branch(branch: str) -> None:
    local = subprocess.run(
        ["git", "show-ref", "--verify", "--quiet", f"refs/heads/{branch}"]
    )
    if local.returncode == 0:
        return
    run("git", "branch", branch, f"origin/{branch}")


def main() -> None:
    root = run("git", "rev-parse", "--show-toplevel", capture=True)
    if not root:
        raise SystemExit("Not inside a Git repository")
    Path(root).resolve()
    require_clean_worktree()
    ensure_filter_repo()

    run("git", "fetch", "origin", "--prune")

    backup_remote = remote_branch_sha(BACKUP_BRANCH)
    print(f"Remote safety backup: {BACKUP_BRANCH} -> {backup_remote[:12]}")

    old_remote = {branch: remote_branch_sha(branch) for branch in BRANCHES}
    for branch in BRANCHES:
        ensure_local_branch(branch)
    old_trees = {
        branch: run("git", "rev-parse", f"{branch}^{{tree}}", capture=True)
        for branch in BRANCHES
    }

    origin_url = run("git", "remote", "get-url", "origin", capture=True)

    callback = r'''
oid = commit.original_id.decode()
subject, sep, body = commit.message.decode("utf-8", "replace").partition("\n")
short = oid[:12]

result_subjects = {
    "27c5996f82e3": "Record V0.2 electrochemistry validation baseline",
    "7baaa6b76276": "Record final V0.2 convergence validation",
    "3b27e6568cd1": "Record V0.3 membrane convergence results",
    "6374e4641ef2": "Record initial V0.4 hydrated-cathode validation",
    "ac13c7a95480": "Record corrected V0.4 proton-sign validation",
    "15f71ef883f7": "Record V0.2/V0.4 quick pipeline validation",
    "0738785f177e": "Record V0.4 quick comparison results",
    "aa6aa4e9d101": "Record corrected V0.4 reference comparison",
    "2978e9ed714b": "Record final V0.4 spatial validation",
    "a38c83596d20": "Record V0.4 RH-voltage sweep results",
    "685d75cb2f70": "Record V0.4 RH sensitivity analysis results",
    "7c2977301a89": "Record final V0.4 RH study and figures",
    "7d870d3c9d9f": "Record V0.5 zero-flux coupling preflight",
    "c4965b382a19": "Record V0.5 nominal coupled solution",
    "08a5986f2a00": "Record V0.5 RH-voltage sweep results",
    "b0e463ec0288": "Record V0.5 quick spatial-convergence results",
    "e3d283e8be88": "Record V0.5 full spatial-convergence results",
    "cb49157df1ba": "Record final V0.5 validation dataset",
    "d40e3a95a075": "Record V0.6 nominal finite-transfer result",
    "4a9c4abe52e0": "Record V0.6 zero-transfer continuity result",
}
special_subjects = {
    "12bb6201c7f6": "Fix Ruff issues in V0.6 coupling code",
    "9438e58526a9": "Document PEMFC research and development history",
}

for prefix, replacement in result_subjects.items():
    if oid.startswith(prefix):
        subject = replacement
        break
for prefix, replacement in special_subjects.items():
    if oid.startswith(prefix):
        subject = replacement
        break

epoch = int(commit.author_date.split()[0])
if epoch <= 1790093967:
    phase = "V0.1"
    phase_desc = "3D oxygen-transport cathode baseline"
elif epoch <= 1790114152:
    phase = "V0.2"
    phase_desc = "coupled cathode electrochemistry and numerical validation"
elif epoch <= 1790114975:
    phase = "V0.3"
    phase_desc = "standalone hydrated-membrane physics"
elif epoch <= 1790167043:
    phase = "V0.4"
    phase_desc = "hydrated cathode, corrected proton balance, and RH study"
elif epoch <= 1790593902:
    phase = "V0.5"
    phase_desc = "partitioned cathode/membrane water coupling"
else:
    phase = "V0.6"
    phase_desc = "finite membrane-to-anode water transfer"

if not subject.startswith("["):
    subject = f"[{phase}] {subject}"

lower = subject.lower()
if "merge" in lower:
    role = "Milestone/integration commit: combines the validated work of this phase."
elif "test" in lower or "validate" in lower:
    role = (
        "Validation commit: constrains the implementation with a physical, numerical, "
        "or regression check."
    )
elif "fix" in lower or "correct" in lower:
    role = (
        "Correction commit: resolves an identified implementation, numerical, "
        "or physical-consistency issue."
    )
elif "result" in lower or "record" in lower:
    role = (
        "Research record: stores reproducible numerical evidence used to accept "
        "or compare this model state."
    )
elif "study" in lower or "sweep" in lower or "convergence" in lower:
    role = (
        "Study infrastructure: makes a controlled numerical experiment reproducible "
        "and restartable."
    )
else:
    role = (
        "Development commit: adds one incremental capability while keeping the model "
        "inspectable."
    )

existing = body.strip()
context = (
    f"Research phase: {phase} — {phase_desc}.\n"
    f"{role}\n"
    "History note: this message was expanded during the research-history rewrite; "
    "the commit tree/content itself was not altered by that rewrite."
)
if existing:
    new_message = subject + "\n\n" + existing + "\n\n" + context + "\n"
else:
    new_message = subject + "\n\n" + context + "\n"
commit.message = new_message.encode()
'''

    refs = [f"refs/heads/{branch}" for branch in BRANCHES]
    run(
        "git",
        "filter-repo",
        "--force",
        "--commit-callback",
        callback,
        "--refs",
        *refs,
    )

    # git-filter-repo may remove origin as a safety measure.
    remotes = run("git", "remote", capture=True).splitlines()
    if "origin" not in remotes:
        run("git", "remote", "add", "origin", origin_url)

    commit_map_path = Path(".git/filter-repo/commit-map")
    mapping: dict[str, str] = {}
    for line in commit_map_path.read_text().splitlines():
        old, new = line.split()
        mapping[old] = new

    def rewritten_sha(prefix: str) -> str:
        matches = [new for old, new in mapping.items() if old.startswith(prefix)]
        if len(matches) != 1:
            raise SystemExit(f"Cannot uniquely resolve rewritten SHA for {prefix}: {matches}")
        return matches[0]

    for tag, (old_prefix, message) in MILESTONES.items():
        subprocess.run(
            ["git", "tag", "-d", tag],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        run("git", "tag", "-a", tag, rewritten_sha(old_prefix), "-m", message.strip())

    # Verify that history rewriting changed commit metadata only, not file trees.
    for branch in BRANCHES:
        new_tree = run("git", "rev-parse", f"{branch}^{{tree}}", capture=True)
        if old_trees[branch] != new_tree:
            raise SystemExit(
                f"Safety check failed: {branch} tip tree changed during rewrite"
            )

    print("\nRewritten history prepared. Force-pushing with explicit leases...")
    for branch in BRANCHES:
        run(
            "git",
            "push",
            f"--force-with-lease=refs/heads/{branch}:{old_remote[branch]}",
            "origin",
            f"{branch}:{branch}",
        )

    run("git", "push", "--force", "origin", *[f"refs/tags/{tag}" for tag in MILESTONES])

    print("\nHistory rewrite complete.")
    print(f"Original remote history remains at origin/{BACKUP_BRANCH}.")
    print("Run: git log --graph --decorate --oneline --all")


if __name__ == "__main__":
    main()
