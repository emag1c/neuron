package neuron

import (
	"fmt"
	"math/rand"
	"testing"
)

func simpleQuadratic(x float64) float64 {
	return ((x*2+x*5+5)*2+6*10+9)*3 + 5
}

func TestNeuron(t *testing.T) {
	conf := &Config{
		Precision:     0.0001,
		Activator:     &LeakyRelu{0.5},
		PreProcessor:  &SumPreProcessor{},
		MaxWeight:     2,
		MaxBias:       2,
	}

	sess := NewSession(0.003)

	inputLayer := []*Neuron{
		NewNeuron(conf, sess.NextIDs(1)[0], rand.Float64(), rand.Float64(), sess),
	}

	// add the input connection
	inputConn := NewConnection(nil, inputLayer[0].ID())
	err := inputLayer[0].AddInputConnections([]*Connection{inputConn})
	if err != nil {
		panic(err)
	}

	// hiddenLayer layer
	hiddenLayer := []*Neuron{
		NewNeuron(conf, sess.NextIDs(1)[0], rand.Float64(), rand.Float64(), sess),
		NewNeuron(conf, sess.NextIDs(1)[0], rand.Float64(), rand.Float64(), sess),
		NewNeuron(conf, sess.NextIDs(1)[0], rand.Float64(), rand.Float64(), sess),
		NewNeuron(conf, sess.NextIDs(1)[0], rand.Float64(), rand.Float64(), sess),
	}

	// connect input to hiddenLayer
	err = ConnectNeurons(inputLayer, hiddenLayer)
	if err != nil {
		panic(err)
	}

	// hiddenLayer layer
	hiddenLayer2 := []*Neuron{
		NewNeuron(conf, sess.NextIDs(1)[0], rand.Float64(), rand.Float64(), sess),
		NewNeuron(conf, sess.NextIDs(1)[0], rand.Float64(), rand.Float64(), sess),
		NewNeuron(conf, sess.NextIDs(1)[0], rand.Float64(), rand.Float64(), sess),
		NewNeuron(conf, sess.NextIDs(1)[0], rand.Float64(), rand.Float64(), sess),
	}

	// connect hidden to hidden
	err = ConnectNeurons(hiddenLayer, hiddenLayer2)
	if err != nil {
		panic(err)
	}

	outputLayer := []*Neuron{
		NewNeuron(conf, sess.NextIDs(1)[0], rand.Float64(), rand.Float64(), sess),
	}

	// connect hiddenLayer to output
	err = ConnectNeurons(hiddenLayer2, outputLayer)
	if err != nil {
		panic(err)
	}

	// add the output connection
	outputConn := NewConnection(outputLayer[0].ID(), nil)
	err = outputLayer[0].AddOutputConnections([]*Connection{outputConn})
	if err != nil {
		panic(err)
	}

	// turn on the neurons
	var neuronCnt float64
	for _, l := range [][]*Neuron{inputLayer, hiddenLayer, hiddenLayer2, outputLayer} {
		for _, n := range l {
			n.On()
			neuronCnt += 1
		}
	}

	fmt.Printf("all %.1f neurons are running\n", neuronCnt)

	EPOCHS := 10
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

		// loop over the dataset
		sess.SetMode(MODE_PREDICTING)
		for i, data := range dataset {
			x := data[0]
			y := data[1]
			inputConn.Forward <- &Packet{
				NeuronID: nil, // first packet has no id
				X:        x,
			}
			out := <-outputConn.Forward // blocking until we have an output
			yhat := out.X
			cost := yhat - y
			costs[i] = cost
			predictions[i] = yhat
			targets[i] = y
			costSum += cost
			pSum += yhat
			tSum += y
		}

		// compute the average error
		avgCost := costSum / float64(DATASET_SIZE)
		avgP := pSum / float64(DATASET_SIZE)
		avgT := tSum / float64(DATASET_SIZE)
		// get the mean absolute error
		sess.SetMode(MODE_TRAINING).
			 SetLoss((avgP - avgT) / neuronCnt)
		inputConn.Forward <- &Packet{} // empty packet that will simply trigger each neuron to update
		// empty the queue
		<-outputConn.Forward
		fmt.Printf("EPOCH: %d TARGET: %.03f PRED: %.03f ERROR: %.03f\n", i+1, avgT, avgP, avgCost)
	}
}

//
//func simpleMultiInputQuadratic(x, y float64) float64 {
//	return ((x*5+5)*y*2+6*10+9)*3 + 5
//}
//
//func TestMultiInputNeuron(t *testing.T) {
//	conf := &Config{
//		LearningRate: 0.003,
//		Precision:    0.0001,
//		Activator:    &LeakyRelu{},
//		PreProcessor: &SumPreProcessor{},
//	}
//
//	sess := NewSession()
//
//	inputLayer := []*Neuron{
//		NewNeuron(conf, sess.NextIDs(1)[0], rand.Float64(), rand.Float64(), sess.ctx),
//		NewNeuron(conf, sess.NextIDs(1)[0], rand.Float64(), rand.Float64(), sess.ctx),
//	}
//
//	// add the input connection
//	inputConns := []*Connection{
//		NewConnection(nil, inputLayer[0].ID()),
//		NewConnection(nil, inputLayer[1].ID()),
//	}
//
//	err := inputLayer[0].AddInputConnections(inputConns)
//	if err != nil {
//		panic(err)
//	}
//
//	// hiddenLayer layer
//	hiddenLayer := []*Neuron{
//		NewNeuron(conf, sess.NextIDs(1)[0], rand.Float64(), rand.Float64(), sess.ctx),
//		NewNeuron(conf, sess.NextIDs(1)[0], rand.Float64(), rand.Float64(), sess.ctx),
//		NewNeuron(conf, sess.NextIDs(1)[0], rand.Float64(), rand.Float64(), sess.ctx),
//		NewNeuron(conf, sess.NextIDs(1)[0], rand.Float64(), rand.Float64(), sess.ctx),
//	}
//
//	// connect input to hiddenLayer
//	err = ConnectNeurons(inputLayer, hiddenLayer)
//	if err != nil {
//		panic(err)
//	}
//
//	// hiddenLayer layer
//	hiddenLayer2 := []*Neuron{
//		NewNeuron(conf, sess.NextIDs(1)[0], rand.Float64(), rand.Float64(), sess.ctx),
//		NewNeuron(conf, sess.NextIDs(1)[0], rand.Float64(), rand.Float64(), sess.ctx),
//		NewNeuron(conf, sess.NextIDs(1)[0], rand.Float64(), rand.Float64(), sess.ctx),
//		NewNeuron(conf, sess.NextIDs(1)[0], rand.Float64(), rand.Float64(), sess.ctx),
//	}
//
//	// connect hidden to hidden
//	err = ConnectNeurons(hiddenLayer, hiddenLayer2)
//	if err != nil {
//		panic(err)
//	}
//
//	outputLayer := []*Neuron{
//		NewNeuron(conf, sess.NextIDs(1)[0], rand.Float64(), rand.Float64(), sess.ctx),
//	}
//
//	// connect hiddenLayer to output
//	err = ConnectNeurons(hiddenLayer2, outputLayer)
//	if err != nil {
//		panic(err)
//	}
//
//	// add the output connection
//	outputConn := NewConnection(outputLayer[0].ID(), nil)
//	err = outputLayer[0].AddOutputConnections([]*Connection{outputConn})
//	if err != nil {
//		panic(err)
//	}
//
//	// turn on the neurons
//	neuron_cnt := 0
//	for _, l := range [][]*Neuron{inputLayer, hiddenLayer, hiddenLayer2, outputLayer} {
//		for _, n := range l {
//			n.On()
//			neuron_cnt += 1
//		}
//	}
//
//	println("all neurons are running")
//
//	EPOCHS := 10000
//	DATASET_SIZE := 100
//
//	// make a dummy dataset
//	dataset := make([][]float64, DATASET_SIZE)
//	for i := 0; i < DATASET_SIZE; i++ {
//		x := rand.Float64()
//		x2 := rand.Float64()
//		dataset[i] = []float64{x, x2, simpleMultiInputQuadratic(x, x2)}
//	}
//
//	for i := 0; i < EPOCHS; i++ {
//
//		targets := make([]float64, DATASET_SIZE)
//		predictions := make([]float64, DATASET_SIZE)
//		costs := make([]float64, DATASET_SIZE)
//		var costSum, pSum, tSum float64
//
//		// loop over the dataset
//		for i, data := range dataset {
//			x := data[0]
//			x2 := data[1]
//			y := data[2]
//			inputConns[0].Forward <- &Packet{
//				NeuronID: nil, // first packet has no id
//				Data:     []float64{x},
//			}
//			inputConns[1].Forward <- &Packet{
//				NeuronID: nil, // first packet has no id
//				Data:     []float64{x2},
//			}
//			out := <-outputConn.Forward // blocking until we have an output
//			cost := out.Data[0] - y
//			costs[i] = cost
//			predictions[i] = out.Data[0]
//			targets[i] = y
//			costSum += cost
//			pSum += out.Data[0]
//			tSum += y
//		}
//
//		// compute the average error
//		avgCost := costSum / float64(DATASET_SIZE)
//		avgP := pSum / float64(DATASET_SIZE)
//		avgT := tSum / float64(DATASET_SIZE)
//
//		// get the mean absolute error
//		outputConn.Backward <- &Packet{
//			NeuronID: nil,                                      // first packet has no id
//			Data:     []float64{avgCost / float64(neuron_cnt)}, // // Data[0] = yhat, p.Data[1] = y
//		}
//		// empty backward queue
//		for _, conn := range inputConns {
//			<-conn.Backward
//		}
//		fmt.Printf("EPOCH: %d TARGET: %.03f PRED: %.03f ERROR: %.03f\n", i+1, avgT, avgP, avgCost)
//	}
//}
