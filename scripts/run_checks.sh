#!/usr/bin/env bash
set -euo pipefail

suite="${1:-fast}"

case "$suite" in
    fast|full) ;;
    *)
        echo "usage: $0 [fast|full]" >&2
        exit 2
        ;;
esac

if ! upstream="$(git rev-parse --abbrev-ref --symbolic-full-name '@{u}' 2>/dev/null)"; then
    echo "check workflow: current branch has no upstream" >&2
    exit 2
fi

git fetch origin

base_head="$(git rev-parse HEAD)"
upstream_head="$(git rev-parse "$upstream")"

if [[ "$base_head" != "$upstream_head" ]]; then
    echo "check workflow: local branch is not synchronized with $upstream" >&2
    echo "run: git pull --ff-only" >&2
    exit 2
fi

if [[ -n "$(git status --porcelain --untracked-files=no)" ]]; then
    echo "check workflow: tracked working tree is not clean" >&2
    echo "commit, restore, or stash tracked changes before make test" >&2
    exit 2
fi

ruff_changed=0
cleanup_on_failure() {
    status=$?
    if (( status != 0 )) && (( ruff_changed != 0 )); then
        git restore --worktree --staged -- src tests scripts
        echo "check workflow: Ruff edits restored after failed validation" >&2
    fi
    exit "$status"
}
trap cleanup_on_failure EXIT

python -m ruff check --fix --unsafe-fixes src tests scripts

if ! git diff --quiet -- src tests scripts; then
    ruff_changed=1
fi

if [[ "$suite" == "full" ]]; then
    OMP_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1         python -m pytest -q -n "${PYTEST_WORKERS:-4}" --dist=load
else
    OMP_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1         python -m pytest -q -n "${PYTEST_WORKERS:-4}" --dist=load -m "not slow"
fi

python -m mypy src tests scripts

if (( ruff_changed == 0 )); then
    echo "Ruff: no changes to commit"
    trap - EXIT
    exit 0
fi

# Re-check the remote immediately before creating a local Ruff commit.
git fetch origin
if [[ "$(git rev-parse HEAD)" != "$(git rev-parse "$upstream")" ]]; then
    git restore --worktree --staged -- src tests scripts
    ruff_changed=0
    echo "check workflow: remote advanced during validation; Ruff edits discarded" >&2
    echo "run: git pull --ff-only && make test" >&2
    trap - EXIT
    exit 2
fi

git add src tests scripts
git commit -m "Ruff fixes"

# Never leave an unpushed automatic Ruff commit behind.
if ! git push; then
    git reset --hard "$base_head"
    ruff_changed=0
    echo "check workflow: push failed; automatic Ruff commit rolled back" >&2
    trap - EXIT
    exit 2
fi

ruff_changed=0
trap - EXIT
echo "Ruff: fixes committed and pushed"
