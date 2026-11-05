#!/bin/bash
# Usage: cutload/run_sweep.sh [OPS [NAME]]
#   OPS as SWEEP_COMPUTE: comma list of case indices or "all" (default 0)
#   NAME: output subdirectory (default sweep)
# Builds the sweep, computes those ops with dumps in cutload/data/NAME/, and renders each dump to
# cutload/out/NAME/P<i>.svg.
set -e
cd "$(dirname "$0")/.."
NAME=${2:-sweep}
D=cutload/data/$NAME
O=cutload/out/$NAME
mkdir -p $D $O
SWEEP_COMPUTE=${1:-0} SWEEP_NAME=$NAME build/bin/FreeCADCmd \
    -c "__file__ = 'cutload/perimeter_sweep.FCMacro'; exec(open(__file__).read())" > $D/run.log 2>&1
grep -E "^P[0-9]+: op state|ADAPTIVE_DUMP: done" $D/run.log
cd cutload
# render the dumps this run wrote (paths as the dumper logged them)
for f in $(grep -o "data/$NAME/P[0-9]*\.[0-9]*\.bin" data/$NAME/run.log | sort -u); do
    case=$(basename "$f" | cut -d. -f1)
    python3 render_svg.py out/$NAME/$case.svg "$f"
done
