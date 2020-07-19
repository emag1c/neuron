package neuron

type Config struct {
	Precision     float64
	Activator     Activator
	PreProcessor  PreProcessor
	RandomFactor  float64
	NumSignals    int
	MaxWeight     float64
	MaxBias       float64
}
