#!/bin/bash
#
# Pre-commit wrapper around scrub-account-numbers.sh.
#
# Pre-commit passes the list of staged files as $@. We scrub each one in
# place, then `git add` the scrubbed versions so the commit picks up the
# clean content. If anything was changed we exit non-zero so the commit
# fails with a clear message — the user re-runs `git commit` and the next
# attempt succeeds because the files are already clean.
#
# Why the failed-commit-on-fix dance: it forces the user to glance at
# `git diff --cached` and confirm the auto-edit is what they wanted before
# the change goes in. Standard pre-commit framework idiom.
#
set -euo pipefail

# Detect account number from AWS CLI (no prompting in a hook).
ACCOUNT=""
if command -v aws >/dev/null 2>&1; then
    ACCOUNT="$(aws sts get-caller-identity --query Account --output text 2>/dev/null || true)"
fi

modified=0
for FILE in "$@"; do
    [ -f "$FILE" ] || continue

    if [ -n "$ACCOUNT" ] && grep -q "$ACCOUNT" "$FILE" 2>/dev/null; then
        # Replace exact account number with placeholder
        sed -i.bak "s/$ACCOUNT/<ACCOUNT_ID>/g" "$FILE"
        rm -f "$FILE.bak"
        git add "$FILE"
        echo "  ✓ Scrubbed exact account in: $FILE"
        modified=$((modified + 1))
        continue
    fi

    # No specific account known (or no match for it): scan for any 12-digit
    # number appearing inside an AWS-ish context.
    if grep -qE '(arn:aws:|s3://|\.amazonaws\.com).*[0-9]{12}' "$FILE" 2>/dev/null; then
        TMP="$(mktemp)"
        sed -E 's/(arn:aws:[^:]*:[^:]*:)[0-9]{12}([:/])/\1<ACCOUNT_ID>\2/g' "$FILE" |
          sed -E 's/(s3:\/\/[^\/]*-)[0-9]{12}(\/)/\1<ACCOUNT_ID>\2/g' |
          sed -E 's/([^0-9])[0-9]{12}(\.amazonaws\.com)/\1<ACCOUNT_ID>\2/g' > "$TMP"
        if ! diff -q "$FILE" "$TMP" >/dev/null 2>&1; then
            mv "$TMP" "$FILE"
            git add "$FILE"
            echo "  ✓ Scrubbed pattern-matched account in: $FILE"
            modified=$((modified + 1))
        else
            rm -f "$TMP"
        fi
    fi
done

if [ "$modified" -gt 0 ]; then
    echo ""
    echo "✗ Pre-commit auto-scrubbed $modified file(s). Review with"
    echo "    git diff --cached"
    echo "  and re-run \`git commit\` to land the cleaned version."
    exit 1
fi

exit 0
