1) Download python
2) clone the project from git
3) cd to the project
4) python.exe -m venv .venv
5) .\.venv\Scripts\pip.exe install opencv-python deepface yt-dlp pillow numpy tensorflow tf-keras
6) .\.venv\Scripts\python.exe .\youtube_downloader.py

25/10 - new option to have a file with all your youtube links, here is the format
1) .\.venv\Scripts\python.exe .\youtube_downloader.py .\youtube_links.txt

*Format 1 (newline-separated):*
```
https://www.youtube.com/watch?v=dQw4w9WgXcQ
https://youtu.be/jNQXAC9IVRw
https://www.youtube.com/watch?v=9bZkp7q19f0
```

*Format 2 (column-separated):*
```
https://www.youtube.com/watch?v=dQw4w9WgXcQ, https://youtu.be/jNQXAC9IVRw
https://www.youtube.com/watch?v=9bZkp7q19f0
