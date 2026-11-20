#!/bin/sh
# Turn a screen recording into review evidence: a frame every N seconds (named by timestamp) and a list of
# stretches where the screen did not change (ffmpeg freezedetect). Usage: video_kit.sh VIDEO OUTDIR [STEP_S=2] [FREEZE_S=3]
set -eu
V=$1; O=$2; STEP=${3:-2}; FRZ=${4:-3}
mkdir -p "$O/frames"
DUR=$(ffprobe -v error -show_entries format=duration -of csv=p=0 "$V")
ffmpeg -v error -y -i "$V" -vf "fps=1/$STEP,scale=1280:-1" "$O/frames/f%04d.png"
i=1
for f in "$O"/frames/f*.png; do  # rename to the second they show: t0000s.png, t0002s.png, ...
  t=$(( (i - 1) * STEP )); mv "$f" "$O/frames/t$(printf %04d "$t")s.png"; i=$((i + 1))
done
freeze() {  # $1 = noise threshold, $2 = heading
  echo "# $2"
  ffmpeg -v info -i "$V" -vf "freezedetect=n=$1:d=$FRZ" -map 0:v -f null - 2>&1 \
    | sed -nE 's/.*freeze_(start|end): *([0-9.]+).*/\1 \2/p' \
    | awk '$1=="start"{s=$2} $1=="end"{printf "%7.1fs -> %7.1fs  (%5.1f s)\n", s, $2, $2-s}'
}
{ freeze 0.00001 "FROZEN: nothing on screen changed at all for >= $FRZ s"
  freeze 0.0002 "NEAR-STATIC: at most a spinner/timer changed for >= $FRZ s (the eye sees no progress)"; } > "$O/freezes.txt"
printf 'video %s: %.1f s, %s frames every %s s in %s/frames\n' "$V" "$DUR" "$(ls "$O"/frames | wc -l | tr -d ' ')" "$STEP" "$O"
cat "$O/freezes.txt"
