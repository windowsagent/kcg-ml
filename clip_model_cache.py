import os
import open_clip

path = os.path.expanduser("~/.cache/clip")
print(f"clip cache path: {path}")

print(os.listdir(path))

