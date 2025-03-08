import yt_dlp
import os
import argparse

def download_video(url, output_path, ydl_opts):
    # Create output path if it doesn't exist
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    ydl_opts['outtmpl'] = f"{output_path}/%(title)s.%(ext)s"
    
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])

def main():
    parser = argparse.ArgumentParser(description="Download YouTube videos.")
    parser.add_argument('url', help='URL of the YouTube video to download')
    parser.add_argument('--output', default='../downloads', help='Output directory for downloaded videos')
    
    args = parser.parse_args()
    
    ydl_opts = {
        'format': 'bestvideo[height<=1080][vcodec^=avc1]+bestaudio/best[height<=1080][vcodec^=avc1]',
        'merge_output_format': 'mp4',
    }
    
    download_video(args.url, args.output, ydl_opts)

if __name__ == "__main__":
    main()