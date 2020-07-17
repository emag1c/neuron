package neuron

import "context"

type Session struct {
	ctx         context.Context
	Stop        context.CancelFunc
	neuronIDcur uint64
}

func (s *Session) NextIDs(cnt int) (ids []NeuronID) {
	ids = make([]NeuronID, cnt)
	for i := 0; i < cnt; i ++ {
		s.neuronIDcur += 1
		ids[i] = NeuronID(s.neuronIDcur)
	}
	return ids
}

func NewSession() *Session {
	ctx, stop := context.WithCancel(context.Background())
	return &Session{
		ctx: ctx,
		Stop: stop,
		neuronIDcur: 0,
	}
}