#!/bin/bash
cd "$(dirname "$0")"

echo "Highlight Reel"
echo "--------------"
echo "Drag a video file into this window and press Enter:"
# plain read (no -r) so the backslash-escapes Terminal adds to dragged
# paths with spaces get unescaped
read video_path

# Strip surrounding quotes and trailing whitespace that paste/drag can add
video_path="${video_path%\'}"
video_path="${video_path#\'}"
video_path="${video_path%\"}"
video_path="${video_path#\"}"
video_path="${video_path%[[:space:]]}"

if [ -z "$video_path" ]; then
	echo "No file given. Exiting."
	exit 1
fi

if [ ! -f "$video_path" ]; then
	echo "File not found: $video_path"
	exit 1
fi

echo ""
echo "Preset [general/sports/talk/gaming] (Enter = general):"
read -r preset

echo "Target reel length in seconds (Enter = config default):"
read -r target

args=()
[ -n "$preset" ] && args+=(--preset "$preset")
[ -n "$target" ] && args+=(--target "$target")

./run.sh "$video_path" "${args[@]}"

stem="$(basename "$video_path")"
stem="${stem%.*}"
if [ -d "outputs/$stem" ]; then
	open "outputs/$stem"
fi

echo ""
echo "Done. Press Enter to close."
read -r
