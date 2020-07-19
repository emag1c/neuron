package neuron

type Packet struct {
	NeuronID *NeuronID
	X        float64
}

type Connection struct {
	ProvidingNeuron *NeuronID
	ConsumingNeuron *NeuronID
	Forward         chan *Packet
}

func NewConnection(provider, consumer *NeuronID) *Connection {
	return &Connection{
		ProvidingNeuron: provider,
		ConsumingNeuron: consumer,
		Forward:         make(chan *Packet, 1),
	}
}
