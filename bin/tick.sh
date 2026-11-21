# Sourced by run.sh and bin/auditgym.js (needs say() and $UV). Keeps the first-run setup quiet but never silent.
# With TICK_FILE and TICK_KB set (a download of known size), the line shows that file's MB of the total and the time left,
# and once, after 30 s, says the network is slow but nothing is stuck.
tick() {  # tick LABEL CMD...: run CMD in the background, show LABEL, seconds and MB until it ends; return CMD's code
  L=$1; shift
  if [ -n "${TICK_FILE:-}" ]; then
    kb() { [ -f "$TICK_FILE" ] && du -sk "$TICK_FILE" 2>/dev/null | awk '{print $1}' || echo 0; }
  else
    D1=$("$UV" cache dir 2>/dev/null); D2=$("$UV" python dir 2>/dev/null)
    mb() { du -sk "$D1" "$D2" 2>/dev/null | awk '{s+=$1} END {print int(s/1024)}'; }
    B=$(mb); BIG=0; [ "${B:-0}" -gt 3000 ] && BIG=1   # a big existing uv cache: du would be slow, show seconds only
  fi
  TK0=$(date +%s); SLOW=0
  "$@" & P=$!
  trap 'kill $P 2>/dev/null; exit 130' INT TERM
  line() {
    S=$(( $(date +%s) - TK0 ))
    if [ -n "${TICK_FILE:-}" ]; then
      K=$(kb); E=""
      [ "$K" -gt 0 ] && [ "$S" -ge 2 ] && [ "$K" -lt "$TICK_KB" ] && E=" · ~$(( S * (TICK_KB - K) / K + 1 )) s left"
      [ "$K" -gt "$TICK_KB" ] && K=$TICK_KB
      echo "$L · $S s · $(( K / 1024 )).$(( K % 1024 * 10 / 1024 )) of $(( TICK_KB / 1024 )).$(( TICK_KB % 1024 * 10 / 1024 )) MB$E"
    elif [ $BIG -eq 0 ]; then echo "$L · $S s · $(( $(mb) - B )) MB on disk so far"; else echo "$L · $S s"; fi
  }
  N=0
  while kill -0 $P 2>/dev/null; do
    sleep 1; N=$((N + 1))
    kill -0 $P 2>/dev/null || break
    if [ -n "${TICK_FILE:-}" ] && [ $SLOW -eq 0 ] && [ $N -ge 30 ]; then  # once: a slow link, not a hang
      SLOW=1; [ -t 2 ] && printf '\r\033[K' >&2
      say "  slower network than usual ($(( $(kb) / N )) KB/s); nothing is stuck, it keeps going (setup can take up to 3 min)"
    fi
    if [ -t 2 ]; then printf '\r\033[K[auditgym]   %s' "$(line)" >&2        # a terminal: one line, redrawn every second
    elif [ $((N % 2)) -eq 0 ]; then say "  $(line)"; fi                      # a log: one line every 2 s
  done
  wait $P; RC=$?
  trap - INT TERM
  if [ -t 2 ]; then printf '\r\033[K' >&2; fi
  [ $RC -eq 0 ] && say "  ✓ $L done in $(( $(date +%s) - TK0 )) s"
  return $RC
}
