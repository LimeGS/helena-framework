#!/bin/sh
# Start the panel over TLS, generating a self-signed certificate the first time.
#
# The panel takes a password and hands back a session cookie. Over plain http
# both cross the network in the clear, and the cookie is a bearer token for a
# service that queues GPU work -- so TLS is not optional here even on a LAN.
#
# Self-signed by default because the alternative is a panel that will not start
# until somebody has arranged a certificate authority, and a deployment step
# that blocks on paperwork is a deployment step people skip. The certificate
# lives in the state directory, so it survives a container rebuild and every
# client that has accepted it once keeps working.
#
# To use a real certificate, set CX_TLS_CERT and CX_TLS_KEY to paths inside the
# container and this script leaves them alone.
set -eu

state="${HELENA_PANEL_TLS_DIR:-/state/tls}"
cert="${CX_TLS_CERT:-$state/panel.crt}"
key="${CX_TLS_KEY:-$state/panel.key}"
port="${HELENA_PANEL_PORT:-8800}"
# Every name and address a browser might use to reach this panel. A certificate
# that covers only "localhost" makes the hostname somebody actually types look
# like an attack.
names="${HELENA_PANEL_TLS_NAMES:-localhost}"

if [ ! -f "$cert" ] || [ ! -f "$key" ]; then
  mkdir -p "$(dirname "$cert")" "$(dirname "$key")"
  alt="DNS:localhost,IP:127.0.0.1,IP:::1"
  for name in $(echo "$names" | tr ',' ' '); do
    case "$name" in
      localhost) ;;
      *[0-9].[0-9]*) alt="$alt,IP:$name" ;;
      *) alt="$alt,DNS:$name" ;;
    esac
  done
  echo "panel: generating a self-signed certificate for $alt"
  openssl req -x509 -newkey rsa:2048 -nodes -days 3650 \
    -keyout "$key" -out "$cert" \
    -subj "/CN=$(echo "$names" | cut -d, -f1)/O=Helena Framework" \
    -addext "subjectAltName=$alt" 2>/dev/null
  chmod 600 "$key"
  echo "panel: fingerprint $(openssl x509 -in "$cert" -noprint_certs -fingerprint -sha256 -noout 2>/dev/null || openssl x509 -in "$cert" -fingerprint -sha256 -noout)"
  echo "panel: it is self-signed, so the first visit warns. Compare that"
  echo "panel: fingerprint rather than clicking through blind."
fi

# Exported so the application knows it is behind TLS and may mark the session
# cookie Secure. A Secure cookie over http is silently dropped by the browser.
CX_TLS_CERT="$cert"
CX_TLS_KEY="$key"
export CX_TLS_CERT CX_TLS_KEY

exec python3 -m uvicorn panel.app:app \
  --host "${HELENA_PANEL_BIND:-0.0.0.0}" --port "$port" \
  --ssl-certfile "$cert" --ssl-keyfile "$key" "$@"
