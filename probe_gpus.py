import json, os, time, urllib.request

KEY = open("/opt/data/.runpod_key").read().strip()
URL = "https://api.runpod.io/graphql"
H = {"Authorization": f"Bearer {KEY}", "Content-Type": "application/json"}
IMAGE = "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04"

# Candidate GPUs, roughly cheapest-compatible first.
# MuJoCo+SB3 needs ~no VRAM; 16GB+ is plenty. Prefer cheap consumer cards.
CANDIDATES = [
    "NVIDIA GeForce RTX 3090",
    "NVIDIA GeForce RTX 3090 Ti",
    "NVIDIA GeForce RTX 4080",
    "NVIDIA GeForce RTX 4080 SUPER",
    "NVIDIA GeForce RTX 4090",
    "NVIDIA GeForce RTX 4070 Ti",
    "NVIDIA RTX A4000",
    "NVIDIA RTX A5000",
    "NVIDIA RTX A6000",
    "NVIDIA L40S",
    "NVIDIA L4",
    "NVIDIA A40",
    "NVIDIA RTX 5000 Ada Generation",
    "NVIDIA RTX 4000 Ada Generation",
]

def q(query):
    req = urllib.request.Request(URL, data=json.dumps({"query": query}).encode(),
                                 headers=H, method="POST")
    return json.loads(urllib.request.urlopen(req, timeout=30).read())

def probe(gpu):
    try:
        d = q(f'mutation {{ podFindAndDeployOnDemand(input: {{ '
              f'name: "probe", imageName: "{IMAGE}", gpuTypeId: "{gpu}", '
              f'containerDiskInGb: 50, volumeInGb: 20, '
              f'ports: "47472/http,22/tcp" }}) {{ id name desiredStatus }} }}')
        p = d.get("data", {}).get("podFindAndDeployOnDemand")
        if p:
            return p["id"]  # available -> got a pod id
        return None, d.get("errors", [{}])[0].get("message", "?")[:60]
    except Exception as e:
        return None, f"ERR {e}"

def terminate(pid):
    try:
        q(f'mutation {{ podTerminate(input: {{ podId: "{pid}" }}) {{ id }} }}')
    except Exception:
        pass

print("Probing live GPU availability (terminating any probe pod)...")
available = []
for gpu in CANDIDATES:
    res = probe(gpu)
    if isinstance(res, str):  # got pod id
        pid = res
        print(f"  AVAILABLE: {gpu} (pod {pid}) -- terminating probe")
        terminate(pid)
        available.append(gpu)
    else:
        _, msg = res
        print(f"  {gpu}: {msg}")
    time.sleep(2)

print("\n=== AVAILABLE NOW ===")
for g in available:
    print(" -", g)
if not available:
    print(" (none right now)")
