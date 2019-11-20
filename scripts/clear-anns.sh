#!/bin/bash

if [ $# -lt 1 ]; then
    CORPUS_PATH=.
else
    CORPUS_PATH="$1"
fi

echo "Clearing all .ann files corresponding to .txt files in $CORPUS_PATH.";
read -p "Are you sure? [Y/N]   " -n 1 -r
echo # New line
if [[ $REPLY =~ ^[Yy]$ ]]; then
    for TXTFILE in $(find "$CORPUS_PATH" -name "*.txt"); do
        echo "Clearing ${TXTFILE%.*}.ann";
        > ${TXTFILE%.*}.ann;
    done
fi
