import subprocess
from pathlib import Path
import re
import itertools
import os

EXE_PATH = Path("out/build/x64-Release/src/")

MODE_LINEAR = 1
MODE_NONLINEAR = 0

def runExperiment(fullPath, mode, resolution, envChanges):

  with open("experiment.conf", "w") as f:
    f.write("10\n") # number of v cycles
    f.write("0\n") # tolerance, not used
    # resolution x, y, z
    for _ in range(3):
      f.write(f"{resolution}\n")
    f.write(f"{mode}\n") # linear / non-linear
    f.write("3\n3\n") # pre and post smoothing
    f.write("0.8\n") # omega
    f.write("1.0\n") # gamma
    f.write("6 -1 -1 -1 -1 -1 -1\n0 1 -1 0 0 0 0\n0 0 0 1 -1 0 0\n0 0 0 0 0 1 -1\n") # stencil
    
  cmd = [fullPath, "experiment.conf"]

  env = os.environ.copy()
  for (k,v) in envChanges.items():
    if k == "PATH":
      env["PATH"] = v + env["PATH"]
    else:
      env[k] = v

  result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True, env=env)

  os.remove("experiment.conf")

  if result.returncode != 0:
    print("eperiment konnte nicht ausgeführt werden")
    print(result.stderr)
    return
  
  pattern = re.compile(r"iter: (\d+) residual: ([\d\.e-]+) Took (\d+)ms")
  matches = pattern.findall(result.stdout)
  if not matches or len(matches) == 0:
    print("Konnte Ergebnisse nicht extrahieren")
    print(result.stdout)
    return
  
  totalTime = 0
  for line in matches:
    iter = int(line[0])
    residual = float(line[1])
    time = int(line[2])

    totalTime += time

  return totalTime / len(matches)

print("\n")

openMpConfigs = [{}, {"OMP_NUM_THREADS": "1"}]
dpcConfig = {"ONEAPI_DEVICE_SELECTOR": "cuda:*", "PATH": r"intel_dpc\llvm\build\bin;"}

impls = [
  (EXE_PATH / "GpuSolve-cpu.exe", openMpConfigs),
  (EXE_PATH / "GpuSolve-gtx.exe", {}),
  (Path("intel_dpc/out.exe"), dpcConfig)
]

modes = [MODE_LINEAR, MODE_NONLINEAR]

resolutions = [63, 127, 255]

# warm up
for impl in impls:
  exe = impl[0]
  envs = impl[1]
  if isinstance(envs, list):
    env = envs[0]
  else:
    env = envs
  print(f"Warmup {exe.name}")
  runExperiment(exe, modes[0], resolutions[0], env)

for (exeTuple, mode, resolution) in itertools.product(impls, modes, resolutions):
  exe = exeTuple[0]
  envs = exeTuple[1]

  printEnv = True
  if not isinstance(envs, list):
    envs = [envs]
    printEnv = False

  modeStr = "NON LINEAR" if mode == 0 else "LINEAR"

  for env in envs:
    avgRun = runExperiment(exe, mode, resolution, env)
    if printEnv:
      print(f"{exe.name} in mode {modeStr} with env {env} and {resolution} points: {avgRun}ms")
    else:
      print(f"{exe.name} in mode {modeStr} and {resolution} points: {avgRun}ms")