# Usage: bash strip_logs.sh <base_directory>
base_dir="$1"
if [ -z "$base_dir" ]; then
  echo "Usage: bash $0 <base_directory>"
  exit 1
fi

# Find all subfolders which are named after numbers
find "$base_dir" -type d -regextype posix-extended -regex '.*/[0-9]+' | while read -r dir; do
  # Skip if doesn't exist (subsubfolders may be already deleted)
  if [ ! -d "$dir" ]; then
    continue
  fi

  echo "Stripping logs in directory: $dir"

  # Delete big folders if they exist
  if [ -d "$dir/memory" ]; then
      rm -rf "$dir/memory"
  fi
  if [ -d "$dir/videos" ]; then
      rm -rf "$dir/videos"
  fi
  if [ -d "$dir/checkpoints" ]; then
      rm -rf "$dir/checkpoints"
  fi
  if [ -d "$dir/models" ]; then
      rm -rf "$dir/models"
  fi
done

echo "Log stripping complete."
