import subprocess
import time
import re
import sys
import os
import webbrowser

print("=" * 60)
print("  STARTING AI TRAFFIC MANAGEMENT LIVE INTERNET TUNNEL")
print("  POWERED BY YOUR LOCAL NVIDIA RTX 3050 GPU")
print("=" * 60)

# 1. Start Streamlit in background
print("\n[1/2] Starting Streamlit Dashboard on port 8501...")
st_proc = subprocess.Popen(
    [sys.executable, "-m", "streamlit", "run", "streamlit_app.py", "--server.port", "8501", "--server.headless", "true"],
    cwd=os.path.dirname(os.path.abspath(__file__))
)

time.sleep(3)

# 2. Start Cloudflare Tunnel
print("[2/2] Generating instant secure HTTPS public link...")
cf_proc = subprocess.Popen(
    ["cloudflared.exe", "tunnel", "--url", "http://localhost:8501"],
    stderr=subprocess.PIPE,
    stdout=subprocess.PIPE,
    text=True,
    bufsize=1,
    cwd=os.path.dirname(os.path.abspath(__file__))
)

url_found = None
start_t = time.time()
while time.time() - start_t < 30:
    line = cf_proc.stderr.readline()
    if not line:
        continue
    match = re.search(r'https://[a-zA-Z0-9-]+\.trycloudflare\.com', line)
    if match:
        url_found = match.group(0)
        break

if url_found:
    print("\n" + "=" * 65)
    print("  SUCCESS! YOUR LIVE PUBLIC LINK IS:")
    print(f"  --> {url_found} <--")
    print("=" * 65 + "\n")
    print("Opening the live link automatically in your web browser...")
    
    link_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "YOUR_LIVE_LINK.txt")
    with open(link_file, "w", encoding="utf-8") as f:
        f.write(f"Traffic Signal AI - Live Public Internet Link:\n\n{url_found}\n\n(Keep this command window open while using the link!)\n")
    
    try:
        webbrowser.open(url_found)
    except Exception:
        pass
    
    print("\nShare this link with anyone! Press Ctrl+C in this window when you want to stop.")
    try:
        cf_proc.wait()
    except KeyboardInterrupt:
        print("\nStopping tunnel...")
        cf_proc.terminate()
        st_proc.terminate()
else:
    print("\nCould not automatically grab URL. Waiting for cloudflared output...")
    try:
        cf_proc.wait()
    except KeyboardInterrupt:
        cf_proc.terminate()
        st_proc.terminate()

