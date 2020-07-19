package neuron

import (
	"fmt"
	"math/rand"
)

func ScaledRand() float64 {
	//r := rand.NormFloat64() * math.Abs(rand.NormFloat64())
	r := rand.NormFloat64()
	fmt.Printf("%.05f\n", r)
	return r
}

// Layer is a simple layer structure that contains a single rank of Neurons
type Layer struct {
	Name    string
	Config  *Config
	Sess    *Session
	Neurons []*Neuron
}

func (l *Layer) On() {
	for _, n := range l.Neurons {
		n.On()
	}
}

func NewLayer(name string, conf *Config, sess *Session, neurons int) *Layer {
	l := &Layer{
		Name:    name,
		Config:  conf,
		Sess:    sess,
		Neurons: make([]*Neuron, neurons),
	}

	ids := sess.NextIDs(neurons)
	for i := 0; i < neurons; i++ {
		l.Neurons[i] = NewNeuron(conf, ids[i], ScaledRand(), ScaledRand(), sess)
	}

	return l
}

 func ConnectLayers(provider, consumer *Layer) error {
 	if provider.Sess != consumer.Sess {
 		return fmt.Errorf("groups must be part of the same session")
	}
	return ConnectNeurons(provider.Neurons, consumer.Neurons)
 }


type InputLayer struct {
	Layer  *Layer
	Inputs []*Connection
}

func NewInputLayer(name string, conf *Config, sess *Session, neurons int) (*InputLayer, error) {
	layer := NewLayer(name, conf ,sess, neurons)
	il := &InputLayer{
		Layer:  layer,
		Inputs: make([]*Connection, len(layer.Neurons)),
	}
	for i, n := range layer.Neurons {
		il.Inputs[i] = NewConnection(nil, n.ID())
		if err := n.AddInputConnections([]*Connection{il.Inputs[i]}); err != nil {
			return nil, err
		}
	}
	return il, nil
}

func (l *InputLayer) Forward(packets ...*Packet) error{
	if len(packets) != len(l.Inputs) {
		return fmt.Errorf("packet count must equal input count")
	}
	for i, in := range l.Inputs {
		in.Forward <- packets[i]
	}
	return nil
}

type OutputLayer struct {
	Layer   *Layer
	Outputs []*Connection
}

func NewOutputLayer(name string, conf *Config, sess *Session, neurons int) (*OutputLayer, error) {
	layer := NewLayer(name, conf ,sess, neurons)
	ol := &OutputLayer{
		Layer:  layer,
		Outputs: make([]*Connection, len(layer.Neurons)),
	}
	for i, n := range layer.Neurons {
		ol.Outputs[i] = NewConnection(n.ID(), nil)
		if err := n.AddOutputConnections([]*Connection{ol.Outputs[i]}); err != nil {
			return nil, err
		}
	}
	return ol, nil
}

func (g *OutputLayer) Forward() []*Packet {
	packets := make([]*Packet, len(g.Outputs))
	for i, in := range g.Outputs {
		packets[i] = <- in.Forward
	}
	return packets
}