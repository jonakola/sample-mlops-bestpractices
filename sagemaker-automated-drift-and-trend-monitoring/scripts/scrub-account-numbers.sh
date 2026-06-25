#!/bin/bash
# Script to scrub AWS account numbers from files
# Replaces actual account numbers with <ACCOUNT_ID> placeholder

set -e

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo "🧹 AWS Account Number Scrubber"
echo "================================"
echo ""

# Get the actual account number from AWS (if available)
ACTUAL_ACCOUNT=""
if command -v aws &> /dev/null; then
    ACTUAL_ACCOUNT=$(aws sts get-caller-identity --query Account --output text 2>/dev/null || echo "")
fi

if [ -z "$ACTUAL_ACCOUNT" ]; then
    echo -e "${YELLOW}⚠${NC} Could not detect AWS account number from AWS CLI"
    echo -e "Please enter your AWS account number (12 digits), or press Enter to search for all patterns:"
    read -r ACTUAL_ACCOUNT
fi

if [ ! -z "$ACTUAL_ACCOUNT" ]; then
    echo -e "${BLUE}Using account number:${NC} $ACTUAL_ACCOUNT"
else
    echo -e "${YELLOW}Searching for any 12-digit numbers in AWS contexts...${NC}"
fi
echo ""

# Files to process (only in git)
FILES_TO_CHECK=$(git ls-files | grep -E '\.(py|ipynb|yaml|yml|md|json|txt|sh|tf|cfg|ini|env\.example)$' || true)

if [ -z "$FILES_TO_CHECK" ]; then
    echo "No files to check"
    exit 0
fi

MODIFIED_FILES=0

# Function to scrub a file
scrub_file() {
    local FILE=$1
    local ACCOUNT=$2
    local TMPFILE=$(mktemp)

    if [ ! -z "$ACCOUNT" ]; then
        # Replace specific account number
        if grep -q "$ACCOUNT" "$FILE" 2>/dev/null; then
            sed "s/$ACCOUNT/<ACCOUNT_ID>/g" "$FILE" > "$TMPFILE"
            mv "$TMPFILE" "$FILE"
            echo -e "${GREEN}✓${NC} Scrubbed: $FILE"
            return 0
        fi
    else
        # Search for any 12-digit number in AWS context and replace
        if grep -qE '(arn:aws:|s3://|\.amazonaws\.com).*[0-9]{12}' "$FILE" 2>/dev/null; then
            # Replace 12-digit numbers that appear in AWS contexts
            sed -E 's/(arn:aws:[^:]*:[^:]*:)[0-9]{12}([:/])/\1<ACCOUNT_ID>\2/g' "$FILE" | \
            sed -E 's/(s3:\/\/[^\/]*-)[0-9]{12}(\/)/\1<ACCOUNT_ID>\2/g' | \
            sed -E 's/([^0-9])[0-9]{12}(\.amazonaws\.com)/\1<ACCOUNT_ID>\2/g' > "$TMPFILE"

            # Check if file actually changed
            if ! diff -q "$FILE" "$TMPFILE" > /dev/null 2>&1; then
                mv "$TMPFILE" "$FILE"
                echo -e "${GREEN}✓${NC} Scrubbed: $FILE"
                return 0
            else
                rm "$TMPFILE"
            fi
        fi
    fi

    return 1
}

echo "Scanning files..."
echo ""

# Process each file
for FILE in $FILES_TO_CHECK; do
    if [ -f "$FILE" ]; then
        if scrub_file "$FILE" "$ACTUAL_ACCOUNT"; then
            ((MODIFIED_FILES++))
        fi
    fi
done

echo ""
echo "================================"
if [ $MODIFIED_FILES -gt 0 ]; then
    echo -e "${GREEN}✓${NC} Scrubbed $MODIFIED_FILES file(s)"
    echo ""
    echo "Next steps:"
    echo "  1. Review the changes: git diff"
    echo "  2. Stage the scrubbed files: git add -u"
    echo "  3. Commit: git commit -m 'Scrub account numbers'"
else
    echo -e "${GREEN}✓${NC} No account numbers found"
fi
echo ""
