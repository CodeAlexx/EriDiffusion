#!/bin/bash
# Clean all Claude references from git history

echo "Removing all Claude references from git history..."

# Reset to original state first
git reset --hard refs/original/refs/heads/master

# Remove Claude references
git filter-branch -f --msg-filter 'perl -pe "
    s/Co-[Aa]uthored-[Bb]y: Claude.*//g;
    s/.*noreply\@anthropic\.com.*//g;
    s/.*🤖.*Claude Code.*//g;
    s/.*Generated with.*Claude Code.*//g;
    s/\[Claude Code\]\(https:\/\/claude\.ai\/code\)//g;
"' -- --all

echo "Done! To push changes: git push --force-with-lease origin master"