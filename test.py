import random
from scipy import optimize
import perceval as pcvl
from perceval.components.unitary_components import PS, BS, PERM
import numpy as np

# Data
n = 4
input = pcvl.BasicState([1]*n)
output_to_max = pcvl.BasicState([n]+[0]*(n-1))
backend = pcvl.BackendFactory.get_backend("SLOS")

# TO-DO: implement a generic circuit of size n with parameters. Code the loss function to maximise the good output. Launch the optimisation procedure. Output the probability and circuit obtained

# We take a universal circuit
circuit = pcvl.GenericInterferometer(n,
    lambda i: BS(theta=pcvl.P(f"theta{i}"),
    phi_tr=pcvl.P(f"phi_tr{i}")),
    phase_shifter_fun_gen=lambda i: PS(phi=pcvl.P(f"phi{i}")))
param_circuit = circuit.get_parameters()
params_init = [random.random()*np.pi for _ in param_circuit]


def loss_function(params):
    for i, value in enumerate(params):
        param_circuit[i].set_value(value)
    backend.set_circuit(circuit)
    backend.set_input_state(input)
    loss = -backend.probability(output_to_max)
    print(loss)
    return loss  # we want to maximise the prob, so we want to minimise the -prob


# We run the otpimisation
o = optimize.minimize(loss_function, params_init, method="L-BFGS-B")

print(f"The maximum probability is {-loss_function(o.x)}")

# For n=4, the probability should be 3/32
# The maximum can also be obtained with the Hadamard matrix :

H4 = (1/2)*np.array([[1,1,1,1], [1,-1,1,-1], [1,1,-1,-1], [1,-1,-1,1]])
backend.set_circuit(pcvl.Unitary(pcvl.Matrix(H4)))
backend.set_input_state(input)
backend.probability(output_to_max)