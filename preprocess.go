package neuron


type PreProcessor interface {
	PreForward(packets map[*NeuronID]*Packet) float64
}

type SumProcessor struct {}

func (p *SumProcessor) PreForward(packets map[*NeuronID]*Packet) float64 {
	var sum float64
	for _, packet := range packets {
		sum += packet.Data[0]
	}
	return sum
}

type AvgProcessor struct {}

func (p *AvgProcessor) PreForward(packets map[*NeuronID]*Packet) float64 {
	var sum float64
	for _, packet := range packets {
		sum += packet.Data[0]
	}
	return sum / float64(len(packets))
}

type MaxProcessor struct {}

func (p *MaxProcessor) PreForward(packets map[*NeuronID]*Packet) float64 {
	var max float64
	for _, packet := range packets {
		if packet.Data[0] > max {
			max = packet.Data[0]
		}
	}
	return max
}