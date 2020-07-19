package neuron

import (
	"fmt"
	"math/rand"
	"testing"
)


func TestLayer(t *testing.T) {
	conf := &Config{
		Precision:    0.0001,
		Activator:    &LeakyRelu{0.5},
		PreProcessor: &SumPreProcessor{},
		MaxBias: 3,
		MaxWeight: 5,
	}

	sess := NewSession(0.003)

	input, err := NewInputLayer("input", conf, sess, 1)
	if err != nil {
		panic(err)
	}

	hidden1 := NewLayer("hidden1", conf, sess, 100)
	if err := ConnectLayers(input.Layer, hidden1); err != nil {
		panic(err)
	}
	hidden2 := NewLayer("hidden2", conf, sess, 500)
	if err := ConnectLayers(hidden1, hidden2); err != nil {
		panic(err)
	}
	output, err := NewOutputLayer("output", conf, sess, 1)
	if err != nil {
		panic(err)
	}
	// connect input to hiddenLayer
	err = ConnectLayers(hidden2, output.Layer)
	if err != nil {
		panic(err)
	}

	// turn on the neurons
	neuronCnt := float64(0)
	for _, l := range []*Layer{input.Layer, hidden1, hidden2, output.Layer} {
		l.On()
		neuronCnt += float64(len(l.Neurons))
	}

	println("all neurons are running")

	EPOCHS := 1000
	DATASET_SIZE := 100

	// make a dummy dataset
	dataset := make([][]float64, DATASET_SIZE)
	for i := 0; i < DATASET_SIZE; i++ {
		x := rand.Float64()
		dataset[i] = []float64{x, simpleQuadratic(x)}
	}

	for i := 0; i < EPOCHS; i++ {

		targets := make([]float64, DATASET_SIZE)
		predictions := make([]float64, DATASET_SIZE)
		costs := make([]float64, DATASET_SIZE)
		var costSum, pSum, tSum float64

		perComplete := float64(i) / float64(EPOCHS)
		sess.SetLearningRate(OneCycleLearningRate(0.001, 0.1, perComplete))
		// loop over the dataset
		sess.SetMode(MODE_PREDICTING)
		for i, data := range dataset {
			x := data[0]
			y := data[1]
			if err := input.Forward(&Packet{
				NeuronID: nil,
				X:        x,
			}); err != nil {
				panic(err)
			}
			out := output.Forward()[0] // blocking until we have an output
			cost := out.X - y
			costs[i] = cost
			predictions[i] = out.X
			targets[i] = y
			costSum += cost
			pSum += out.X
			tSum += y
		}

		// compute the average error
		avgCost := costSum / float64(DATASET_SIZE)
		avgP := pSum / float64(DATASET_SIZE)
		avgT := tSum / float64(DATASET_SIZE)
		// get the mean absolute error
		sess.SetMode(MODE_TRAINING).SetLoss((avgP - avgT) / neuronCnt)

		if err := input.Forward(&Packet{}); err != nil {
			panic(err)
		}
		// empty the queue
		output.Forward()
		fmt.Printf("EPOCH: %d [%.04f%%] LR: %.04f TARGET: %.03f PRED: %.03f ERROR: %.03f\n",
			i+1, perComplete * 100, sess.LearningRate(), avgT, avgP, avgCost)
	}
}
