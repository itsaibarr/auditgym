# Sourced by run.sh and bin/auditgym.js (needs say() and $UV). Keeps the first-run setup quiet but never silent.
tick() {  # tick LABEL CMD...: run CMD in the background; every 2 s say LABEL, seconds and MB fetched so far; return CMD's code
  L=$1; shift
  D1=$("$UV" cache dir 2>/dev/null); D2=$("$UV" python dir 2>/dev/null)
  mb() { du -sk "$D1" "$D2" 2>/dev/null | awk '{s+=$1} END {print int(s/1024)}'; }
  B=$(mb); BIG=0; [ "${B:-0}" -gt 3000 ] && BIG=1   # a big existing uv cache: du would be slow, show seconds only
  "$@" & P=$!
  trap 'kill $P 2>/dev/null; exit 130' INT TERM
  N=0
  while kill -0 $P 2>/dev/null; do
    sleep 1; N=$((N + 1))
    if [ $((N % 2)) -eq 0 ] && kill -0 $P 2>/dev/null; then
      if [ $BIG -eq 0 ]; then say "  $L · $N s · $(( $(mb) - B )) MB so far"; else say "  $L · $N s"; fi
    fi
  done
  trap - INT TERM
  wait $P
}
