package neuron


type PreProcessor interface {
	PreProcess(values []float64) float64
}

type SumPreProcessor struct {}

func (p *SumPreProcessor) PreProcess(values []float64) float64 {
	var (
		sum float64
		cnt int
	)
	for _, v := range values {
		sum += v
		cnt++
	}
	return sum
}
