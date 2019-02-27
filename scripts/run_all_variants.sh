#!/bin/bash
# Requires the Unix task spooler (package task-spooler in Ubuntu).
# Expects to be run from the lstm-causality-tagger root directory.

SEED=1341932933
OUT_DIR=$HOME/Documents/Work/Research/Causality/lstm-causality-tagger/runs
LOG_DIR=$OUT_DIR
DATA_ROOT_DIR=/var/www/brat/data/BECauSE/corpus
BASE_CMD="build/jdunietz-opt-debug/lstm-causality/lstm-causality-tagger --cnn-mem 800 --cnn-seed $SEED --train -f 20 -e 5 -C"

tsp -S 4

# Columns:
# Run_type data_dir extra_flags
read -r -d '' PER_RUN_VARS << EOF
vanilla         $DATA_ROOT_DIR
no-new-vectors  $DATA_ROOT_DIR     --word-dim=0
no-parse-paths  $DATA_ROOT_DIR     --parse-path-hidden-dim=0
no-action-hist  $DATA_ROOT_DIR     --action-dim=0
known-conns     $DATA_ROOT_DIR     -K
oracle-conns    $DATA_ROOT_DIR     -o
EOF

run_pipeline() {
	NAME=$1
	DATA_DIR=$2
	FLAGS=$3
    echo "Run type:" $NAME
    tsp -n -L "lstm-$NAME" bash -c "$BASE_CMD -r $DATA_DIR $FLAGS > '$OUT_DIR/$NAME.out' 2> '$LOG_DIR/$NAME.log'"
    sleep 15s
}

mkdir -p $OUT_DIR
mkdir -p $LOG_DIR

printf '%s\n' "$PER_RUN_VARS" | while IFS="\n" read line; do
    read RUN_TYPE DIR FLAGS <<<$line
    run_pipeline $RUN_TYPE $DIR $FLAGS
done

tsp -l # Print spooled tasks
