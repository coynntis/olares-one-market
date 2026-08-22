#!/usr/bin/env bash
# Scaffold Olares voice app from omnivoiceone template.
set -euo pipefail
APP="${1:?usage: scaffold-voice-app.sh APPNAME SRVNAME CLINAME}"
SRV="${2:?}"
CLI="${3:?}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SRC="$ROOT/omnivoiceone"
DST="$ROOT/$APP"

if [[ -d "$DST" ]]; then
  echo "exists: $DST" >&2
  exit 1
fi

cp -R "$SRC" "$DST"

# rename subcharts
mv "$DST/omnivoiceone" "$DST/$CLI"
mv "$DST/omnivoiceonesrv" "$DST/$SRV"

replace() {
  local f="$1"
  [[ -f "$f" ]] || return 0
  sed -i '' \
    -e "s/omnivoiceonesrv/$SRV/g" \
    -e "s/omnivoiceonecli/${CLI}cli/g" \
    -e "s/omnivoiceone/$APP/g" \
    -e "s/OmniVoice TTS One/$APP/g" \
    -e "s/OmniVoice/$APP/g" \
    "$f"
}

while IFS= read -r f; do replace "$f"; done < <(find "$DST" -type f)

echo "scaffolded $DST (edit OlaresManifest + deployment)"
