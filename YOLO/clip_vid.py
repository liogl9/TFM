from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip

ffmpeg_extract_subclip("whitenoise0.mp4", 3,
                       128, targetname="test.mp4")
