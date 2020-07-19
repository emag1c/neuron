package neuron

import (
	"math/rand"
)

type RandProvider interface {
	RandNew() float64
}

type RandNormal struct {
	Scale float64
}

func (r *RandNormal) RandNew() float64 {
	return rand.NormFloat64() * r.Scale
}
