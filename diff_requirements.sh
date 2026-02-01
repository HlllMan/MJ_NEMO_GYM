#!/bin/bash

# Usage: ./diff_requirements.sh [requirements_file]
# Uses pip's resolver via --dry-run for accurate diff detection
# Better for building dev docker images - shows exactly what pip would do

REQUIREMENTS_FILE=${1:-requirements.txt}

if [ ! -f "$REQUIREMENTS_FILE" ]; then
    echo "Error: Requirements file '$REQUIREMENTS_FILE' not found"
    exit 1
fi

# Create temp file for output
TEMP_OUTPUT=$(mktemp)

echo ""
echo "============================================================"
echo "Package Diff: $REQUIREMENTS_FILE vs Installed (pip resolver)"
echo "============================================================"

# Run pip install --dry-run and capture output
pip install --dry-run -r "$REQUIREMENTS_FILE" 2>&1 | tee "$TEMP_OUTPUT"

echo ""
echo "============================================================"
echo "SUMMARY (filtered)"
echo "============================================================"

# Count different types
new_packages=$(grep -c "^Collecting" "$TEMP_OUTPUT" 2>/dev/null || echo 0)
satisfied=$(grep -c "Requirement already satisfied" "$TEMP_OUTPUT" 2>/dev/null || echo 0)
would_install=$(grep "Would install" "$TEMP_OUTPUT" 2>/dev/null || echo "")
uninstalling=$(grep -i "uninstalling" "$TEMP_OUTPUT" 2>/dev/null || echo "")

echo ""
echo "--- NEW PACKAGES (Collecting) ---"
grep "^Collecting" "$TEMP_OUTPUT" | sed 's/Collecting /  /' || echo "  (none)"

echo ""
echo "--- VERSION CHANGES (Uninstalling) ---"
if [ -n "$uninstalling" ]; then
    echo "$uninstalling" | sed 's/^/  /'
else
    echo "  (none)"
fi

echo ""
echo "--- WOULD INSTALL ---"
if [ -n "$would_install" ]; then
    echo "$would_install" | sed 's/Would install/  Would install:/'
else
    echo "  (none - all requirements satisfied)"
fi

echo ""
echo "------------------------------------------------------------"
echo "Stats: $new_packages to collect | $satisfied already satisfied"
if [ -n "$uninstalling" ]; then
    echo "WARNING: Some packages would be MODIFIED (see Uninstalling above)"
fi
echo "------------------------------------------------------------"

rm -f "$TEMP_OUTPUT"
