import subprocess
from pathlib import Path
import re
import itertools
import os

EXE_PATH = Path("out/build/x64-Release/src/")

MODE_LINEAR = 0
MODE_NONLINEAR = 1
MODE_NEWTON = 2

def runExperiment(fullPath, mode, resolution, envChanges):

  with open("experiment.conf", "w") as f:
    f.write("10\n") # number of v cycles
    f.write("10e-3\n") # tolerance
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
    print(result.stdout)
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

  ramUsage = []
  pattern = re.compile(r"Current ram usage: (\d+)\s")
  matches = pattern.findall(result.stdout)
  for line in matches:
    ram = int(line)
    ramUsage.append(ram / 1024 / 1024)

  return (totalTime, ramUsage)

print("\n")

openMpConfigs = [{}, {"OMP_NUM_THREADS": "1"}]
dpcConfig = {"ONEAPI_DEVICE_SELECTOR": "cuda:*", "PATH": r"intel_dpc\llvm\build\bin;"}

impls = [
  (EXE_PATH / "GpuSolve-cpu.exe", openMpConfigs),
  (EXE_PATH / "GpuSolve-gtx.exe", {}),
  (Path("intel_dpc/out.exe"), dpcConfig)
]

modes = [MODE_LINEAR, MODE_NONLINEAR, MODE_NEWTON]

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

savedResults = {}
for (exeTuple, mode, resolution) in itertools.product(impls, modes, resolutions):
  exe = exeTuple[0]
  envs = exeTuple[1]

  printEnv = True
  if not isinstance(envs, list):
    envs = [envs]
    printEnv = False

  modeStr = ""
  if mode == MODE_LINEAR:
    modeStr = "LINEAR"
  elif mode == MODE_NONLINEAR:
    modeStr = "NON LINEAR"
  else:
    modeStr = "NEWTON"

  for env in envs:
    avgRun, ramUsage = runExperiment(exe, mode, resolution, env)
    if printEnv:
      print(f"{exe.name} in mode {modeStr} with env {env} and {resolution} points: {avgRun}ms")
    else:
      print(f"{exe.name} in mode {modeStr} and {resolution} points: {avgRun}ms")
    #for ram in ramUsage:
    #  print(f"\tRAM usage: {ram:.2f}MiB")

    key = f"{exe.name}_{mode}_{resolution}_{env}"
    savedResults[key] = {"avgRun": avgRun, "ramUsage": ramUsage}

print("")

# avg time
for resolution in resolutions:
  print(f"Results for resolution {resolution}:")
  for exeTuple in impls:
    
    envs = exeTuple[1]
    if not isinstance(envs, list):
      envs = [envs]

    for env in envs:
      outStr = "\\addplot coordinates {"
      for mode in modes:
        resultKey = f"{exeTuple[0].name}_{mode}_{resolution}_{env}"
        result = savedResults[resultKey]

        modeStr = ""
        if mode == MODE_LINEAR:
          modeStr = "lin"
        elif mode == MODE_NONLINEAR:
          modeStr = "non"
        else:
          modeStr = "newton"

        outStr += f"({modeStr},{result['avgRun']}) "

      outStr += "}; %" + exeTuple[0].name + " " + str(env)
      print(outStr)

print("")

# ram usage
for resolution in resolutions:
  for mode in modes:

    modeStr = ""
    if mode == MODE_LINEAR:
      modeStr = "lin"
    elif mode == MODE_NONLINEAR:
      modeStr = "non"
    else:
      modeStr = "newton"

    print(f"RAM usage for resolution {resolution} and problem {modeStr}:")
    outStr = ""
    for exeTuple in impls:

      envs = exeTuple[1]
      if not isinstance(envs, list):
        envs = [envs]

      for env in envs:
        resultKey = f"{exeTuple[0].name}_{mode}_{resolution}_{env}"
        result = savedResults[resultKey]

        outStr += "\\addplot coordinates { "
        for (i, ram) in enumerate(result['ramUsage']):
          outStr += f"({i+1},{ram:.2f}) "
        
        outStr += "};\n"
        outStr += "\\addlegendentry{" +exeTuple[0].name + "}; %" + str(env) + "\n"

    print(outStr)