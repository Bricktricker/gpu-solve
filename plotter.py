import numpy as np
import matplotlib.pyplot as plt

def u(x, y, z):
    return (x - x*x) * (y - y*y) * (z - z*z)

def readFile(path):
    with open(path, "r") as f:
        headerParts = f.readline().split(" ")
        xDim = int(headerParts[0])
        yDim = int(headerParts[1])
        zDim = int(headerParts[2])
        mesh = np.zeros((xDim, yDim, zDim))

        for line in f:
            parts = line.split(" ")
            xIdx = int(parts[0])
            yIdx = int(parts[1])
            zIdx = int(parts[2])
            value = float(parts[3])
            mesh[xIdx, yIdx, zIdx] = value

    return mesh

def truePlot(n):
    X, Y, Z = np.meshgrid(np.linspace(0.0, 1.0, n), np.linspace(0.0, 1.0, n), np.linspace(0.0, 1.0, n))
    output = u(X, Y, Z)
    return output

computed = readFile("out/build/x64-Debug/src/v_6.txt")
n = computed.shape[0]

trueOut = truePlot(n)
spliceTrue = trueOut[:, 65, 65]
spliceComp = computed[:, 65, 65]

line = np.linspace(0.0, 1.0, n)

plt.plot(line, spliceTrue, label="true curve")
plt.plot(line, spliceComp, label="computed curve")
plt.title("Result")
plt.legend()
plt.show()
