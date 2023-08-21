# how to actually create trainable quantum circuit 
# use "ansatz" = parameterized circuit run on our default state

from qiskit.circuit.library import NLocal, CCXGate, CRZGate, RXGate
from qiskit.circuit import Parameter

theta = Parameter("θ")
ansatz = NLocal(
    num_qubits=5,
    rotation_blocks=[RXGate(theta), CRZGate(theta)],
    entanglement_blocks=CCXGate(),
    entanglement=[ [0, 1, 2], [0, 2, 3], [4, 2, 1], [3, 1, 0] ],
    reps=2,
    insert_barriers=True,
)
ansatz.decompose().draw("mpl")

