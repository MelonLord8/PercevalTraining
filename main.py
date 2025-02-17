import perceval as pcvl
from perceval.algorithm import Sampler
from math import pi
from perceval.components import BS, PS
import numpy as np
import scipy.optimize
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Generate dataset with 2 features
X, y = make_classification(n_samples=100, n_features=2, n_classes=2, n_clusters_per_class=1, n_redundant = 0, random_state=42)

# Split into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

CNOT = pcvl.catalog["heralded cnot"].build_processor()
def phaseGate(param):
    return pcvl.Circuit(2).add(1, PS(param))
theta = [pcvl.P(f"theta{i}") for i in range(6)]
x = [pcvl.P(f"x{i}") for i in range(3)]

feature_map = pcvl.Processor(pcvl.SLOSBackend(), 4)
ansatz = pcvl.Processor(pcvl.SLOSBackend(), 4)

depth = 2

for i in range(2*depth + 1):
    if i % 2 == 0:
        ansatz.add(0, phaseGate(theta[i]))
        ansatz.add(2, phaseGate(theta[i + 1])) 
    else:
        ansatz.add(0,CNOT)

feature_map.add(0, BS.H())
feature_map.add(1, BS.H())
feature_map.add(0, phaseGate(x[0]))
feature_map.add(1, phaseGate(x[1]))
feature_map.add(0, CNOT)
feature_map.add(0, phaseGate(x[2]))
feature_map.add(0, CNOT)

vqc = pcvl.Processor(pcvl.SLOSBackend(), 4)

vqc.add(0, feature_map)
vqc.add(0, ansatz)

init_state = pcvl.BasicState([1, 0, 1, 0])
vqc.with_input(init_state)

def get_probs(data):
    x[0].set_value(2*data[0])
    x[1].set_value(2*data[1])
    x[2].set_value(2*(pi - data[0])*(pi - data[1]))
    sampler = Sampler(vqc)
    probs = sampler.probs()['results'] 
    return probs[pcvl.BasicState([1, 0, 1, 0])] + probs[pcvl.BasicState([1, 0, 0, 1])]

def loss(params):
    for i in range(len(params)):
        theta[i].set_value(params[i])
    out = np.array([get_probs(X_train[i]) for i in range(X_train.shape[0])])
    loss = -np.mean(y_train * np.log2(out) + (1 - y_train) * np.log2(1 - out))
    return loss

init_params = np.random.random(6)
def callback(intermediate_result: scipy.optimize.OptimizeResult):
    print(intermediate_result.fun)
res = scipy.optimize.minimize(loss, init_params, method = "L-BFGS-B", callback = callback)