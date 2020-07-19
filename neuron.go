package neuron

import (
	"fmt"
	"math"
	"math/rand"
	"sync"
)

type NeuronID uint64

type Neuron struct {
	Conf    *Config
	id      NeuronID
	Inputs  []*Connection
	Outputs []*Connection
	cache   *NeuronCache
	weight  float64
	bias    float64
	session *Session
	mu      sync.Mutex
	pre     PreProcessor
	alive   bool
}

func (n *Neuron) ID() *NeuronID {
	return &n.id
}

func (n *Neuron) Forward() {
	//inputsExpected := len(n.Inputs)
	packets := map[*NeuronID]*Packet{}

	var (
		x  float64
		xs []float64
	)

	for _, conn := range n.Inputs {
		packet := <-conn.Forward
		packets[packet.NeuronID] = packet
	}

	if n.session.Predicting() { // we are predicting
		xs = make([]float64, len(packets))
		i := 0
		for _, p := range packets {
			xs[i] = p.X
			i++
		}
		x = n.Conf.PreProcessor.PreProcess(xs)
		z := x*n.weight + n.bias
		a := n.Conf.Activator.Forward(z)
		n.cache.Add(x, z, a)
		x = a
		//fmt.Printf(">+< %v x = %.3f, z = %.3f\n", n.ID(), x, z)
	}
	if n.session.Training() { // we are updating weights and biases
		// update weights and bias if we were passed a loss value
		n.UpdateWeightAndBias()
	}
	// send packet up the chain to all connected neurons
	for _, conn := range n.Outputs {
		conn.Forward <- &Packet{
			NeuronID: &n.id,
			X:        x,
		}
	}
}

func (n *Neuron) UpdateWeightAndBias() {
	n.mu.Lock()
	defer n.mu.Unlock()
	defer n.cache.Zero()
	// get random factors for weight and bias updates
	wRandFactor := float64(1)
	bRandFactor := float64(1)
	if n.Conf.RandomFactor > 0 {
		wRandFactor += n.Conf.RandomFactor * rand.NormFloat64()
		bRandFactor += n.Conf.RandomFactor * rand.NormFloat64()
	}
	// get cached values values
	cached := n.cache.Get()
	var zSum, z float64
	for _, c := range cached {
		zSum += c[1]
	}
	// average the a and z values from this neuron's memory
	z = zSum / float64(len(cached))
	d := float64(1)
	if z == 0 {
		// prevent 0 values
		z = 1e-10
	}
	loss := n.session.Loss()
	wLoss := loss * (z / (z + n.bias)) // the weight's portion of the loss
	bLoss := loss - wLoss              // the bias' portion of the loss
	// update the weight and bias
	wNew := (d*wLoss*math.Abs(wLoss) * n.session.LearningRate() + n.weight) * wRandFactor
	bNew := (d*bLoss*math.Abs(bLoss) * n.session.LearningRate() + n.bias) * bRandFactor

	if n.Conf.MaxWeight != 0 {
		if wNew > n.Conf.MaxWeight {
			wNew = n.Conf.MaxWeight
		} else if wNew < -n.Conf.MaxWeight {
			wNew = -n.Conf.MaxWeight
		}
	}
	if n.Conf.MaxBias != 0 {
		if bNew > n.Conf.MaxBias {
			bNew = n.Conf.MaxBias
		} else if bNew < -n.Conf.MaxBias {
			bNew = -n.Conf.MaxBias
		}
	}
	//fmt.Printf("wOld: %.3f bOld: %.3f wRand: %.3f bRand: %.3f\n", n.weight, n.bias, wRandFactor, bRandFactor)
	//fmt.Printf("loss: %.3f wNew: %.3f bNew: %.3f wLoss: %.3f bLoss: %.3f\n", loss, wNew, bNew, wLoss, bLoss)

	if math.Abs(wNew-n.weight) > n.Conf.Precision {
		n.weight = wNew
	}
	if math.Abs(bNew-n.bias) > n.Conf.Precision {
		n.bias = bNew
	}
}

func (n *Neuron) AddInputConnections(conn []*Connection) error {
	n.mu.Lock()
	defer n.mu.Unlock()
	if n.alive {
		return fmt.Errorf("cannot add connections to active neuron")
	}
	n.Inputs = append(n.Inputs, conn...)
	return nil
}

func (n *Neuron) AddOutputConnections(conn []*Connection) error {
	n.mu.Lock()
	defer n.mu.Unlock()
	if n.alive {
		return fmt.Errorf("cannot add connections to active neuron")
	}
	n.Outputs = append(n.Outputs, conn...)
	return nil
}

func (n *Neuron) MainLoop() {
	for {
		// keep looping until context is done
		select {
		case <-n.session.ctx.Done(): // stop on context cancel
			n.mu.Lock()
			n.alive = false
			n.mu.Unlock()
			return
		default: // keep running
		}
		n.Forward()
	}
}

func (n *Neuron) On() {
	n.mu.Lock()
	n.alive = true
	n.mu.Unlock()
	// start the values and backwards loops in their own routines
	// this allows backward backwards propagation and values propagation to occur at the same time
	go n.MainLoop()
}

func NewNeuron(conf *Config, id NeuronID, weight, bias float64, sess *Session) *Neuron {
	return &Neuron{
		Conf:    conf,
		id:      id,
		Inputs:  []*Connection{},
		Outputs: []*Connection{},
		cache: &NeuronCache{
			values: [][]float64{},
			mu:     sync.RWMutex{},
		},
		bias:    bias,
		weight:  weight,
		session: sess,
		mu:      sync.Mutex{},
		alive:   false,
	}
}

func ConnectNeurons(providers []*Neuron, consumers []*Neuron) error {
	for _, provider := range providers {
		for _, consumer := range consumers {
			conn := NewConnection(provider.ID(), consumer.ID())
			if err := provider.AddOutputConnections([]*Connection{conn}); err != nil {
				return err
			}
			if err := consumer.AddInputConnections([]*Connection{conn}); err != nil {
				return err
			}
		}
	}
	return nil
}
