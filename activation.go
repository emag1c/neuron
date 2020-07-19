package neuron

import "math"

type Activator interface {
	Forward(input float64) float64
	Backward(input float64) float64
}

type Relu struct{}

func (r *Relu) Forward(input float64) float64 {
	return math.Max(0, input)
}

func (r *Relu) Backward(input float64) float64 {
	if input > 0 {
		return 1
	}
	return 0
}

type LeakyRelu struct {
	Alpha float64
}

func (r *LeakyRelu) Forward(input float64) float64 {
	if input < 0 {
		return r.Alpha * input
	}
	return input
}

func (r *LeakyRelu) Backward(input float64) float64 {
	if input >= 0 {
		return 1
	}
	return r.Alpha
}


