# Sourced by run.sh and bin/auditgym.js (needs say() and $UV). Keeps the first-run setup quiet but never silent.
tick() {  # tick LABEL CMD...: run CMD in the background, show LABEL, seconds and MB on disk until it ends; return CMD's code
  L=$1; shift
  D1=$("$UV" cache dir 2>/dev/null); D2=$("$UV" python dir 2>/dev/null)
  mb() { du -sk "$D1" "$D2" 2>/dev/null | awk '{s+=$1} END {print int(s/1024)}'; }
  B=$(mb); BIG=0; [ "${B:-0}" -gt 3000 ] && BIG=1   # a big existing uv cache: du would be slow, show seconds only
  TK0=$(date +%s)
  "$@" & P=$!
  trap 'kill $P 2>/dev/null; exit 130' INT TERM
  line() { S=$(( $(date +%s) - TK0 )); if [ $BIG -eq 0 ]; then echo "$L · $S s · $(( $(mb) - B )) MB on disk so far"; else echo "$L · $S s"; fi; }
  N=0
  while kill -0 $P 2>/dev/null; do
    sleep 1; N=$((N + 1))
    kill -0 $P 2>/dev/null || break
    if [ -t 2 ]; then printf '\r\033[K[auditgym]   %s' "$(line)" >&2        # a terminal: one line, redrawn every second
    elif [ $((N % 2)) -eq 0 ]; then say "  $(line)"; fi                      # a log: one line every 2 s
  done
  wait $P; RC=$?
  trap - INT TERM
  if [ -t 2 ]; then printf '\r\033[K' >&2; fi
  [ $RC -eq 0 ] && say "  ✓ $L done in $(( $(date +%s) - TK0 )) s"
  return $RC
}
