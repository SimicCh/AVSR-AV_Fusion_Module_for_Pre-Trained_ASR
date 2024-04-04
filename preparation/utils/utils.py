from moviepy.config import FFMPEG_BINARY
from moviepy.tools import subprocess_call


def extract_audio_from_video(vid_fname, outfile, samplerate):
    cmd = [
            FFMPEG_BINARY,
            "-i",
            vid_fname,
            "-f", "wav",
            "-ac", str(1),
            "-ar", str(samplerate),
            "-vn", "-y",
            outfile,
            "-loglevel", "quiet"
        ]
    subprocess_call(cmd, logger=None)
    
    return True