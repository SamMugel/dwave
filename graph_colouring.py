import dimod

# use the exact solver to find energies for all states. This is only
# realistic for very small problems.

exactsolver = dimod.ExactSolver()

# Set up the QUBO. Start with the equations from the slides:
# - red - green - blue + 2 * red * green + 2 * red * blue + 2 * blue * green
# and remember the order: (red, green, blue)

pre = 1  # lagrange multiplier

# compute coefficients
vec = [1, 2, 3, 4, 5]
diag_terms = []
off_diag_terms = []
normr = [0]
for i in vec:
    value = pre * (i ** 2) - pre * 16 * i - 5
    diag_terms.append(value)
    off_diag_terms.append([])
    if abs(value) > normr[0]:
        normr[0] = abs(value)
    for j in vec:
        value = 2 + 2 * pre * i * j
        off_diag_terms[i - 1].append(value)
        if abs(value) > normr[0]:
            normr[0] = abs(value)

# normalize coefficients
rescaled_diag_terms = []
rescaled_off_diag_terms = []
for i in vec:
    rescaled_diag_terms.append(diag_terms[i - 1] / normr[0])
    rescaled_off_diag_terms.append([])
    for j in vec:
        rescaled_off_diag_terms[i - 1].append(off_diag_terms[i - 1][j - 1] / normr[0])

# compute QUBO
Q = {}
for i in vec:
    Q.update({(i, i): rescaled_diag_terms[i - 1]})
    for j in vec:
        if j > i and j != i:  # do not double count!!
            Q.update({(i, j): rescaled_off_diag_terms[i - 1][j - 1]})

print(Q)

# There's no need for a constant, so we can use exactsolver directly.
results = exactsolver.sample_qubo(Q)

# print the results
for smpl, energy in results.data(["sample", "energy"]):
    print(smpl, energy)


# running it on the Qc is easy, load the token, the url,
# load the quantum solver and that's it.
