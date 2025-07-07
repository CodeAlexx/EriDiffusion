#!/bin/bash
# Script to remove Claude co-author from git history

echo "This will rewrite git history to remove Claude co-author lines"
echo "WARNING: This will change commit hashes!"
echo ""
read -p "Continue? (y/n) " -n 1 -r
echo ""

if [[ $REPLY =~ ^[Yy]$ ]]; then
    # Use git filter-branch to remove Co-authored-by lines with Claude
    git filter-branch --msg-filter '
        sed "/Co-Authored-By: Claude/d; /Co-authored-by: Claude/d; /noreply@anthropic.com/d"
    ' -- --all
    
    echo ""
    echo "Git history rewritten. To push changes:"
    echo "git push --force-with-lease origin master"
    echo ""
    echo "WARNING: Force pushing will overwrite remote history!"
else
    echo "Cancelled"
fi