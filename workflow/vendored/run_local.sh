#!/bin/bash
# Local stand-in for the `sbatch --wait ...` job prefix used by workflow/vendored/*.smk.
# Accepts the same arguments the rules pass to sbatch, honours -D, -o (with %j) and --wrap,
# ignores scheduler-only flags, and runs the wrapped command in the foreground.
set -euo pipefail
dir=. log=/dev/stdout cmd=
while [[ $# -gt 0 ]]; do
    case "$1" in
        -D) dir=$2; shift 2 ;;
        -o) log=$2; shift 2 ;;
        --wrap=*) cmd=${1#--wrap=}; shift ;;
        --wrap) cmd=$2; shift 2 ;;
        -A|-p|-t|-c|-J|--mem|--gres|--qos|-N|--exclude) shift 2 ;;
        *) shift ;;
    esac
done
log=${log//%j/local}
mkdir -p "$(dirname "$log")"
cd "$dir"
bash -c "$cmd" > "$log" 2>&1
