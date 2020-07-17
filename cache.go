package neurons

import "sync"

type NeuronCache struct {
	forward  [][]float64 // forward cache
	backward [][]float64 // backwards cache
	mu       sync.RWMutex
}

func (pc *NeuronCache) GetForward() [][]float64 {
	pc.mu.RLock()
	defer pc.mu.RUnlock()
	return pc.forward
}

func (pc *NeuronCache) AddForward(v ...float64) {
	pc.mu.Lock()
	defer pc.mu.Unlock()
	pc.forward = append(pc.forward, v)
}

func (pc *NeuronCache) GetBackward() [][]float64 {
	pc.mu.RLock()
	defer pc.mu.RUnlock()
	return pc.backward
}

func (pc *NeuronCache) AddBackward(v ...float64) {
	pc.mu.Lock()
	defer pc.mu.Unlock()
	pc.backward = append(pc.backward, v)
}

func (pc *NeuronCache) ZeroForward() {
	pc.mu.Lock()
	defer pc.mu.Unlock()
	pc.forward = [][]float64{}
}

func (pc *NeuronCache) ZeroBackward() {
	pc.mu.Lock()
	defer pc.mu.Unlock()
	pc.backward = [][]float64{}
}