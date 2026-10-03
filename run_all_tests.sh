#!/bin/bash
# Runs every test file sequentially, one log per file, then a summary.
export JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64
export PATH=$JAVA_HOME/bin:$PATH
cd "$(dirname "$0")"
mkdir -p /tmp/testlogs && rm -f /tmp/testlogs/*
for f in tests/test_*.py; do
  n=$(basename $f .py)
  python3 -m pytest -p no:cacheprovider -q "$f" > /tmp/testlogs/$n.log 2>&1
  echo "$n exit=$?" >> /tmp/testlogs/_summary.txt
done
echo DONE >> /tmp/testlogs/_summary.txt
