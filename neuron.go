package neuron

import (
	"context"
	"fmt"
	"math"
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
	ctx     context.Context
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

	for _, conn := range n.Inputs {
		rcv := <-conn.Forward
		packets[rcv.NeuronID] = rcv
	}
	// do all required calculations over received packets
	x := n.Conf.PreProcessor.PreForward(packets)
	z := x * n.weight + n.bias
	a := n.Conf.Activator.Forward(z)
	// todo: implement hook here
	n.cache.AddForward(x, z, a)
	// send packet up the chain to all connected neurons
	for _, conn := range n.Outputs {
		conn.Forward <- &Packet{
			NeuronID: &n.id,
			Data: []float64{a},
		}
	}
}

func (n *Neuron) Backwards() {

	packets := map[*NeuronID]*Packet{}

	for _, conn := range n.Outputs {
		rcv := <-conn.Backward
		packets[rcv.NeuronID] = rcv
	}

	n.mu.Lock()
	defer n.mu.Unlock()

	var (
		//nyhat float64
		//nyhatSum float64 // this neuron's pred
		lossSum float64
		zSum float64
	)
	// loop over packets to get sums
	for _, p := range packets {
		// p.Data[0] = yhat
		// p.Data[1] = y
		lossSum += p.Data[0]
	}
	loss := lossSum / float64(len(packets))
	// get cached forward values
	cached := n.cache.GetForward()
	// loop over cached values
	for _, v := range cached {
		// v[0] = xSum  (sum of inputs before applying weight and bias)
		// v[1] = z  (pre-activation)
		// v[2] = a  (post-activation)
		zSum += v[1]
		//nyhatSum += v[2]
	}
	// clear the forward cache
	n.cache.ZeroForward()
	// get this neuron's average pred
	//nyhat = nyhatSum / float64(len(cached))
	//nyhat = zSum / float64(len(cached))
	//theta := nyhat * n.Conf.Activator.Backward(nyhat)  // activator derivative
	wLoss := loss * (zSum / (zSum + n.bias))  // the weight's portion of the loss
	bLoss := loss - wLoss  // the bias' portion of the loss
	// update the weight and bias
	n.weight -= wLoss * math.Abs(wLoss) * n.Conf.LearningRate
	n.bias -= bLoss * math.Abs(bLoss) * n.Conf.LearningRate
	//fmt.Printf(">+< %v theta: %.03f loss: %.03f wLoss: %.03f weight: %.03f bias: %.03f\n", &n,
	//	theta, loss, wLoss, n.weight, n.bias)


	// get each input's value based on ratio
	for _, conn := range n.Inputs {
		conn.Backward <- &Packet{
			NeuronID: &n.id,
			Data:     []float64{loss},  // Data[0] = yhat, p.Data[1] = y
		}
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

func (n *Neuron) ForwardLoop() {
	for {
		// keep looping until context is done
		select {
		case <-n.ctx.Done(): // stop on context cancel
			n.mu.Lock()
			n.alive = false
			n.mu.Unlock()
			return
		default: // keep running
		}
		n.Forward()
	}
}

func (n *Neuron) BackwardLoop() {
	for {
		// keep looping until context is done
		select {
		case <-n.ctx.Done(): // stop on context cancel
			n.mu.Lock()
			n.alive = false
			n.mu.Unlock()
			return
		default: // keep running
		}
		n.Backwards()
	}
}

func (n *Neuron) On() {
	n.mu.Lock()
	n.alive = true
	n.mu.Unlock()
	// start the forward and backwards loops in their own routines
	// this allows backward backwards propagation and forward propagation to occur at the same time
	go n.ForwardLoop()
	go n.BackwardLoop()
}

func NewNeuron(conf *Config, id NeuronID, weight, bias float64, ctx context.Context) *Neuron {
	return &Neuron{
		Conf:    conf,
		id:      id,
		Inputs:  []*Connection{},
		Outputs: []*Connection{},
		cache:       &NeuronCache{
			forward:  [][]float64{},
			backward: [][]float64{},
			mu:       sync.RWMutex{},
		},
		bias:        bias,
		weight:      weight,
		ctx:         ctx,
		mu:          sync.Mutex{},
		alive:       false,
	}
}


func ConnectNeurons(providers []*Neuron, consumers []*Neuron) error {
	for _, provider := range providers {
		for _, consumer := range consumers {
			conn := NewConnection(provider.ID(), consumer.ID(), 0)
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
