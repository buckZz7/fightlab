#!/usr/bin/env python3
"""Spin a RunPod pod, render punch_3d.mp4 from traj.npz, pull it back."""
import json, sys, time, urllib.request

KEY = open("/opt/data/.runpod_key").read().strip()
HDRS = {"Content-Type": "application/json", "Authorization": f"Bearer {KEY}",
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0 Safari/537.36",
        "Accept": "application/json", "Origin": "https://www.runpod.io", "Referer": "https://www.runpod.io/"}

def gql(q):
    req = urllib.request.Request("https://api.runpod.io/graphql",
        data=json.dumps({"query": q}).encode(), headers=HDRS, method="POST")
    with urllib.request.urlopen(req) as r:
        return json.loads(r.read())

cmd = sys.argv[1] if len(sys.argv) > 1 else "status"

if cmd == "launch":
    # Match the original training pod config exactly (no explicit ports
    # field — RunPod auto-opens SSH on the PyTorch image).
    mut = """mutation { podFindAndDeployOnDemand(input: {
      name: "fightlab-render",
      imageName: "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04",
      gpuTypeId: "NVIDIA GeForce RTX 3090",
      gpuCount: 1,
      volumeInGb: 20,
      containerDiskInGb: 20,
      minVcpuCount: 8,
      minMemoryInGb: 30
    }) { id desiredStatus costPerHr } }"""
    print(json.dumps(gql(mut), indent=2))
elif cmd == "status":
    print(json.dumps(gql('query { myself { pods { id name desiredStatus runtime { uptimeInSeconds ports { ip publicPort type } } } } }'), indent=2))
elif cmd == "stop":
    print(json.dumps(gql(f'mutation {{ podTerminate(input: {{podId: "{sys.argv[2]}"}}) }}'), indent=2))
