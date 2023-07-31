import torch
import torch.nn as nn 

Transition = namedtuple('Transition', 
('state', 'action', 'next_state', 'reward'))


# class QuantumAgent: 
#     def __init__(self): 
#         self.a_opt = tf.keras.optimizers.Adam(learning_rate=7e-3)
#         self.c_opt = tf.keras.optimizers.Adam(learning_rate=7e-3)
#         self.actor = actor()
#         self.critic = critic()
#         self.clip_pram = 0.2

class DQN(nn.Module): 
    def __init__(self, n_observations, n_actions): 
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, n_actions)

    def forward(self, x): 
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)





# we define one quantum layer as a rotation specified by wx, wy, wz parameters
# in each respective axis 
def layer(W): 
    for i in range(n_qubit): 
        qml.RX(W[i, 0], wires = i)
        qml.RY(W[i, 1], wires = i)
        qml.RZ(W[i, 2], wires = i)


class V(nn.Module):
    def __init__(self):
        super(V, self).__init__()
        self.fc1 = nn.Linear(4,256)
        self.fc_v = nn.Linear(256,1)
    def forward(self,x):
        x = F.relu(self.fc1(x))
        v = self.fc_v(x)
        return v




# Use for parameterization 
@qml.qnode(dev, interface = 'torch')
def circuit(W, s): 

    for i in range(n_qubit): 
        qml.RY(np.pi*s[i], wires = i)

    layer(W[0])

    for i in range(n_qubit - 1): 
        qml.CNOT(wires = [i, i+1])
    layer(W[1])

    for i in range(n_qubit - 1): 
        qml.CNOT(wires = [i, i+1])

    layer(W[2])

    for i in range(n_qubit - 1): 
        qml.CNOT(wires = [i, i+1])
    layer(W[3])
    qml.CNOT(wires = [0, 2])
    qml.CNOT(wires = [1, 3])


    return [qml.expval(qml.PauliY(ind)) for ind in range(2, 4)]


W = Variable(torch.DoubleTensor(np.random.rand(4, 4, 3)), requires_grad = True)

v = V()
quantum_circuit = circuit 
optim1 = optim.Adam([W], lr = 1e-3)
optim2 = optim.Adam(v.parameters(), lr = 1e-5)





