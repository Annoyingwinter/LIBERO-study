#!/usr/bin/env bash
# Get the LIBERO-Spatial demonstrations this repo trains on.
#
# Only task 0 is used:
#   libero_spatial / pick_up_the_black_bowl_between_the_plate_and_the_ramekin
#   and_place_it_on_the_plate_demo.hdf5        (~485 MB, 50 demos)
#
# Full suite is 10 files / ~7 GB and is not needed for any number in the README.
#
# Two routes are offered because neither is strictly better:
#
#   (A) LIBERO's own downloader  -- authoritative, version-matched, but it pulls
#       the entire libero_spatial suite in one go, so it costs ~7 GB.
#   (B) a single-file fetch      -- 485 MB, but the URL is assembled by hand.
#
# Route (A) is the default so that the default path is the one LIBERO actually
# supports. Note the URL in route (B) was derived by reading LIBERO's own
# downloader (libero/libero/utils/download_utils.py: HF_REPO_ID =
# "yifengzhu-hf/LIBERO-datasets", allow_patterns="libero_spatial/*") and was NOT
# verified with a live request, because the machine this repo was assembled on
# had no outbound network at that point. Treat it as unverified.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="${DATA_DIR:-$REPO/data/libero_spatial}"
FILE="pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate_demo.hdf5"
LIBERO_DIR="${LIBERO_DIR:-$REPO/LIBERO}"

if [ "${ROUTE:-A}" = "A" ]; then
  echo "route A: LIBERO official downloader (downloads the whole suite, ~7 GB)"
  echo
  if [ ! -d "$LIBERO_DIR" ]; then
    echo "LIBERO checkout not found at $LIBERO_DIR"
    echo "  git clone --branch main https://github.com/Lifelong-Robot-Learning/LIBERO.git \\"
    echo "      $LIBERO_DIR"
    echo "  pip install -e $LIBERO_DIR"
    exit 1
  fi
  # libero_spatial is one of four suites; --datasets selects it.
  ( cd "$LIBERO_DIR" && python benchmark_scripts/download_libero_datasets.py \
      --datasets libero_spatial --use-huggingface --download-dir "$REPO/data" )
  SRC="$REPO/data/libero_spatial/$FILE"
else
  echo "route B: single-file fetch (485 MB)"
  BASE_URL="${BASE_URL:-https://huggingface.co/datasets/yifengzhu-hf/LIBERO-datasets/resolve/main/libero_spatial}"
  mkdir -p "$DATA_DIR"
  SRC="$DATA_DIR/$FILE"
  if [ -f "$SRC" ]; then
    echo "already present: $SRC ($(du -h "$SRC" | cut -f1))"
  else
    # -C - resumes a partial transfer; --retry survives the flaky link that
    # originally cost this project a week of blocked downloads.
    curl -L -C - --retry 10 --retry-delay 5 --retry-all-errors -o "$SRC" "$BASE_URL/$FILE"
  fi
fi

echo
echo "dataset ready: $SRC"
echo
echo "Next step -- add the 'next_obs' group that robomimic's dataset loader"
echo "requires and the released demos do not contain:"
echo
echo "  python scripts/prepare_data.py \\"
echo "      --in  '$SRC' \\"
echo "      --out '$DATA_DIR/task0_nextobs.hdf5'"