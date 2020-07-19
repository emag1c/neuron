package neuron

import "sync"

type NeuronCache struct {
	values [][]float64 // values cache
	mu     sync.RWMutex
}

func (pc *NeuronCache) Get() [][]float64 {
	pc.mu.RLock()
	defer pc.mu.RUnlock()
	return pc.values
}

func (pc *NeuronCache) Add(v ...float64) {
	pc.mu.Lock()
	defer pc.mu.Unlock()
	pc.values = append(pc.values, v)
}

func (pc *NeuronCache) Zero() {
	pc.mu.Lock()
	defer pc.mu.Unlock()
	pc.values = [][]float64{}
}