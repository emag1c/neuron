package neuron

type Config struct {
	LearningRate float64
	Precision    float64
	Activator    Activator
	PreProcessor PreProcessor
}