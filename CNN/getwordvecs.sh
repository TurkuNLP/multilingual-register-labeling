#!/bin/bash

# Download word vectors.

set -euo pipefail

SCRIPT="$(basename "$0")"
# https://stackoverflow.com/a/246128
SCRIPTDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
WVDIR="$SCRIPTDIR/wordvecs"

mkdir -p "$WVDIR"

BASEURL="https://dl.fbaipublicfiles.com/arrival/vectors"
LANGUAGES="sv"  #"fi en"

for lang in $LANGUAGES; do
    path="$WVDIR/wiki.multi.$lang.vec"
    if [ -s "$path" ]; then
	echo "$path exists, not downloading again" 2>&1
    else
	echo "downloading word vectors for $lang" 2>&1
	wget "$BASEURL/wiki.multi.$lang.vec" -O "$path"
    fi
done
