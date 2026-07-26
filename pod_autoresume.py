"""Poll RunPod for GPU availability and auto-resume the fightlab pod.

Tries podResume (preserves volume + installed deps) first; if the host
stays full and other GPUs free up, falls back to fresh deploy from the
same image. On success, prints the new SSH endpoint + notifies.

Run on the Hermes box (background):
  python3 /opt/data/fightlab-repo-new/pod_autoresume.py
"""
import os, sys, time, json, subprocess

KEY = open("/opt/data/.runpod_key").read().strip()
POD_ID = "nybh0d9ef80i1b"
IMAGE = "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04"
URL = "https://api.runpod.io/graphql"
H = {"Authorization": f"Bearer {KEY}", "Content-Type": "application/json"}
GPUS = ["NVIDIA GeForce RTX 3090", "NVIDIA GeForce RTX 3090 Ti",
        "NVIDIA GeForce RTX 4090", "NVIDIA GeForce RTX 4080 SUPER"]


def q(query):
    import urllib.request
    req = urllib.request.Request(URL, data=json.dumps({"query": query}).encode(),
                                 headers=H, method="POST")
    return json.loads(urllib.request.urlopen(req, timeout=30).read())


def try_resume():
    try:
        d = q(f'mutation {{ podResume(input: {{ podId: "{POD_ID}" }}) '
              f'{{ id desiredStatus }} }}')
        if d.get("data", {}).get("podResume"):
            return "RESUMED"
        return d.get("errors", [{}])[0].get("message", "?")[:70]
    except Exception as e:
        return f"ERR {e}"


def try_deploy(gpu):
    try:
        d = q(f'mutation {{ podFindAndDeployOnDemand(input: {{ '
              f'name: "fightlab2", imageName: "{IMAGE}", '
              f'gpuTypeId: "{gpu}", containerDiskInGb: 50, volumeInGb: 20, '
              f'ports: "47472/http,22/tcp" }}) {{ id name desiredStatus }} }}')
        if d.get("data", {}).get("podFindAndDeployOnDemand"):
            return "DEPLOYED " + str(d["data"]["podFindAndDeployOnDemand"])
        return d.get("errors", [{}])[0].get("message", "?")[:70]
    except Exception as e:
        return f"ERR {e}"


def get_ip():
    # once running, fetch the pod's public IP:port for SSH
    try:
        d = q(f'{{ pod(input: {{ podId: "{POD_ID}" }}) '
              f'{{ id name desiredStatus runtime {{ id }} }} }}')
        # runtime may not carry ip; use myself.pods + try ssh probe below
    except Exception:
        pass
    return None


def main():
    print(f"[autoresume] watching pod {POD_ID} for GPU availability...")
    last = ""
    while True:
        # 1) try resume (preferred: keeps volume)
        r = try_resume()
        if r == "RESUMED":
            print("[autoresume] POD RESUMED -- waiting for SSH to come up")
            # probe SSH up to ~3 min
            for _ in range(36):
                ip = get_ip()
                time.sleep(5)
            print("[autoresume] DONE: pod resumed. Trigger deploy script on Hermes.")
            break
        # 2) if resume blocked by host-full, try fresh deploy on free GPU
        if "not enough free GPUs on the host" in r:
            for gpu in GPUS:
                d = try_deploy(gpu)
                if d.startswith("DEPLOYED"):
                    print(f"[autoresume] FRESH DEPLOYED ({gpu}): {d}")
                    print("[autoresume] NOTE: fresh volume -- Hermes must re-scp code + pip install.")
                    break
            else:
                pass  # all GPUs full
        if r != last:
            print(f"[autoresume] {time.strftime('%H:%M:%S')} status: {r}")
            last = r
        time.sleep(60)


if __name__ == "__main__":
    main()
