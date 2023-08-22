# TO create a reference state |p> from a default state |0>, we run the default state through a non-parameterized unitary

from qiskit import QuantumCircuit 

# Basic referece unitary Ur = X0
qc = QuantumCircuit(3)
qc.x(0)
qc.draw("mpl")


# Template circuits for reference unitary 
from qiskit.circuit.library import TwoLocal 
from math import pi 


# x rotation gate followed by controlled-z gate 

reference_circuit = TwoLocal(2, "rx", "cz", entanglement = "linear", reps = 1)
theta_list = [pi/2, pi/3, pi/3, pi/2]

from qiskit_nature.second_q.circuit.library import HartreeFock
from qiskit_nature.second_q.mappers import JordanWignerMapper

num_spatial_orbitals = 2
num_particles = (1, 1)

mapper = JordanWignerMapper()

h2_reference_state = HartreeFock(
    num_spatial_orbitals=num_spatial_orbitals,
    num_particles=num_particles,
    qubit_mapper=mapper,
)

h2_reference_state.decompose().draw("mpl")

from qiskit.circuit.library import ZZFeatureMap

# zz feature map can be used to encode clasical data 

data = [0.1, 0.2]

zz_feature_map_reference = ZZFeatureMap(feature_dimension=2, reps=2)
zz_feature_map_reference = zz_feature_map_reference.bind_parameters(data)
zz_feature_map_reference.decompose().draw("mpl")









