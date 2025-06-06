import subprocess

# Replace with your Pi 5's IP address
PI5_IP = "192.168.4.1"
PORT = 5000

# Command: capture from camera and stream to Pi 5 over UDP
command = f"""
libcamera-vid -t 0 --width 1240 --height 720 --framerate 24 --codec h264 -o - |
ffmpeg -f h264 -i - -f mpegts udp://{PI5_IP}:{PORT}
"""

print(f"Starting stream to {PI5_IP}:{PORT}")
subprocess.run(command, shell=True)
