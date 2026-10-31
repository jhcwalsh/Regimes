#!/bin/bash
# ------------------------------------------------------------------------
# Hourly health check of regimes.lazyeconomist.com -- Mac Mini only.
#
# Two checks:
#   1. the public URL answers through the Cloudflare tunnel (/_stcore/health)
#   2. scripts/healthcheck.py inside the container: the data loads as the app
#      loads it (refreshing a cache older than a week) and every series is recent
#
# Pushes to ntfy when the site first goes wrong, again every REMIND_HOURS while
# it stays wrong, and once when it recovers. A healthy run writes one log line.
# The topic lives in ~/apps/regimes/.health.env (NTFY_TOPIC=...), untracked.
#
# Manual run:  ~/apps/regimes/scripts/check_regimes.sh
# Scheduled:   ~/Library/LaunchAgents/com.lazyeconomist.regimes.health.plist
#              (from deploy/; hourly)
# ------------------------------------------------------------------------
set -u

APP_DIR="${APP_DIR:-$HOME/apps/regimes}"
TASK="health"
LOG_DIR="$APP_DIR/logs"
LOG="$LOG_DIR/$TASK.log"
LOCK="$LOG_DIR/$TASK.lock"
STATE="$LOG_DIR/$TASK.state"   # "ok", or "fail <epoch of the last push>"
ENV_FILE="$APP_DIR/.health.env"
URL="https://regimes.lazyeconomist.com"
REMIND_HOURS=24

# Absolute paths: launchd does not source .zprofile. /usr/local/bin/docker is
# OrbStack's symlink. Overridable only for testing against stubs.
DOCKER="${DOCKER:-/usr/local/bin/docker}"
CURL="${CURL:-/usr/bin/curl}"

mkdir -p "$LOG_DIR"
stamp() { date -u +%FT%TZ; }

# Keep the log small: an hourly job otherwise grows it forever.
if [ -f "$LOG" ] && [ "$(wc -c < "$LOG")" -gt 1000000 ]; then
    tail -n 2000 "$LOG" > "$LOG.tmp" && mv "$LOG.tmp" "$LOG"
fi

push() {
    # $1 = title, $2 = priority, $3 = tags, $4 = body
    local topic=""
    if [ -f "$ENV_FILE" ]; then
        topic=$(sed -n 's/^NTFY_TOPIC=//p' "$ENV_FILE" | tr -d "\"' \r")
    fi
    if [ -z "$topic" ]; then
        echo "[$(stamp)] no NTFY_TOPIC in $ENV_FILE; push not sent" >> "$LOG"
        return 0
    fi
    if ! "$CURL" -s -o /dev/null -m 20 \
        -H "Title: $1" -H "Priority: $2" -H "Tags: $3" \
        --data-binary "$4" "https://ntfy.sh/$topic"; then
        echo "[$(stamp)] ntfy push failed" >> "$LOG"
    fi
}

if ! mkdir "$LOCK" 2>/dev/null; then
    echo "[$(stamp)] $TASK already running, exiting" >> "$LOG"
    exit 0
fi
trap 'rmdir "$LOCK" 2>/dev/null' EXIT

problems=""

# 1. Public URL. Three tries a minute apart, so a tunnel blip is not an alert.
public_ok=0
for attempt in 1 2 3; do
    if [ "$("$CURL" -s -m 20 "$URL/_stcore/health")" = "ok" ]; then
        public_ok=1
        break
    fi
    [ "$attempt" -lt 3 ] && sleep 60
done
if [ "$public_ok" -eq 0 ]; then
    problems="$URL does not answer (tunnel or container down)"
fi

# 2. Data, inside the container. Skipped when the container is not running.
if [ -z "$problems" ]; then
    out=$(cd "$APP_DIR" && "$DOCKER" compose exec -T regimes python scripts/healthcheck.py 2>&1)
    rc=$?
    echo "$out" | grep -v use_container_width >> "$LOG"
    if [ "$rc" -ne 0 ]; then
        problems=$(echo "$out" | grep -v use_container_width | tail -n 15)
    fi
fi

prev=$(cat "$STATE" 2>/dev/null || echo ok)
now=$(date +%s)

if [ -z "$problems" ]; then
    echo "[$(stamp)] ok" >> "$LOG"
    if [ "${prev%% *}" = "fail" ]; then
        push "Regimes is healthy again" "default" "white_check_mark" \
            "$(printf 'regimes.lazyeconomist.com passed its health check at %s.\n%s\n' "$(stamp)" "$URL")"
    fi
    echo ok > "$STATE"
    exit 0
fi

echo "[$(stamp)] FAIL: $problems" >> "$LOG"
last_push=0
if [ "${prev%% *}" = "fail" ]; then
    last_push=${prev#fail }
fi
if [ "${prev%% *}" != "fail" ] || [ $((now - last_push)) -ge $((REMIND_HOURS * 3600)) ]; then
    title="Regimes site has a problem"
    [ "${prev%% *}" = "fail" ] && title="Regimes site still has a problem"
    push "$title" "high" "warning" \
        "$(printf '%s\n\nChecked %s. Log: %s\n' "$problems" "$(stamp)" "$LOG")"
    last_push=$now
fi
echo "fail $last_push" > "$STATE"
exit 1
