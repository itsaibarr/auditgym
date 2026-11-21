import subprocess, os, socket
print("nvidia-smi:", subprocess.run("nvidia-smi -L", shell=True, capture_output=True, text=True).stdout or "none", flush=True)
print("dev:", [d for d in os.listdir("/dev") if "nvidia" in d])
try:
    socket.getaddrinfo("huggingface.co", 443); print("internet: yes")
except Exception as e:
    print("internet: no", e)
