#!/bin/bash
# Clean up old task output files to free /tmp space
find /tmp/claude-12189 -name "*.output" -not -name "bv2isdytk.output" -exec truncate -s 0 {} + 2>/dev/null
echo "Cleaned up"
df -h /tmp
