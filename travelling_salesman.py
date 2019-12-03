import time
import numpy as np
from hybrid.reference.kerberos import KerberosSampler
from dwave.system.samplers import DWaveSampler

N = 48
# city to city distances
D = np.random.randint(10, size=(N, N))
# Tunable parameters.
gamma = 8500
chainstrength = 4500
numruns = 100
print(
    "Tunable parameters: \n\tGamma: \t\t\t",
    gamma,
    "\n\tChain Strength: \t",
    chainstrength,
    "\n\tNumber of runs: \t",
    numruns,
)

Q = {}
for i in range(N * N):
    for j in range(N * N):
        Q[(i, j)] = 0

print("Q matrix with", len(Q), "entries created.")


# Function to compute index in Q for variable x_(a,b)
def x(a, b):
    return a * N + b


for v in range(N):
    for j in range(N):
        Q[(x(v, j), x(v, j))] += -1 * gamma
        for k in range(j + 1, N):
            Q[(x(v, j), x(v, k))] += 2 * gamma

print("Added", N, "row constraints to Q matrix.")

for j in range(N):
    for v in range(N):
        Q[(x(v, j), x(v, j))] += -1 * gamma
        for w in range(v + 1, N):
            Q[(x(v, j), x(w, j))] += 2 * gamma

print("Added", N, "column constraints to Q matrix.")

for u in range(N):
    for v in range(N):
        if u != v:
            for j in range(N):
                Q[(x(u, j), x(v, (j + 1) % N))] += D[u][v]

print("Objective function added.")

Q = {k: v for k, v in Q.items() if v != 0}
print("Q matrix reduced to", len(Q), "entries.")

start = time.time()
resp = KerberosSampler().sample_qubo(Q)
end = time.time()
time_KS = end - start

print("KerberosSampler call complete using", time_KS, "seconds.")

start = time.time()
# First solution is the lowest energy solution found
sample = next(iter(resp))

# Display energy for best solution found
print("Energy: ", next(iter(resp.data())).energy)

# Print route for solution found
route = [-1] * N
for node in sample:
    if sample[node] > 0:
        j = node % N
        v = (node - j) / N
        if route[j] != -1:
            print("Stop " + str(i) + " used more than once.\n")
        route[j] = int(v)

# Compute and display total mileage
mileage = 0
for i in range(N):
    mileage += D[route[i]][route[(i + 1) % N]]
mileage_KS = mileage
print("Mileage: ", mileage_KS)

end = time.time()
print("Total processing time: ", end - start)

print("\nRoute:\n")
for i in range(N):
    if route[i] != -1:
        print(str(i) + ":  " + cities[route[i]] + "," + states[route[i]] + "\n")
    else:
        print(str(i) + ":  No city assigned.\n")
