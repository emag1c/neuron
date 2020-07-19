package neuron

import (
	"context"
	"sync"
	"time"
)

type Mode string

const (
	MODE_PREDICTING = Mode("predicting")
	MODE_TRAINING   = Mode("training")
	MODE_DUAL       = Mode("dual")
	MODE_OFF        = Mode("")
)

type SessionCache struct {
	loss        map[time.Time]float64
	predictions map[time.Time]float64
	accuracy    map[time.Time]float64
	// todo: add more metrics
}

type Session struct {
	ctx          context.Context
	Stop         context.CancelFunc
	neuronIDcur  uint64
	mode         Mode
	loss         float64
	learningRate float64
	mu           sync.RWMutex
}

func (s *Session) Ctx() context.Context {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return s.ctx
}

func (s *Session) SetLearningRate(lr float64) *Session {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.learningRate = lr
	return s
}

func (s *Session) LearningRate() float64 {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return s.learningRate
}

func (s *Session) Predicting() bool {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return s.mode == MODE_PREDICTING || s.mode == MODE_DUAL
}

func (s *Session) Training() bool {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return s.mode == MODE_TRAINING || s.mode == MODE_DUAL
}

func (s *Session) SetMode(m Mode) *Session {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.mode = m
	return s
}

func (s *Session) Mode() Mode {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return s.mode
}

func (s *Session) SetLoss(loss float64) *Session {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.loss = loss
	return s
}

func (s *Session) Loss() float64 {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return s.loss
}

func (s *Session) NextIDs(cnt int) (ids []NeuronID) {
	s.mu.Lock()
	defer s.mu.Unlock()
	ids = make([]NeuronID, cnt)
	for i := 0; i < cnt; i++ {
		s.neuronIDcur += 1
		ids[i] = NeuronID(s.neuronIDcur)
	}
	return ids
}

func NewSession(lr float64) *Session {
	ctx, stop := context.WithCancel(context.Background())
	return &Session{
		ctx:          ctx,
		Stop:         stop,
		mode:         MODE_OFF,
		mu:           sync.RWMutex{},
		learningRate: lr,
	}
}


func OneCycleLearningRate(lrMin, lrMax float64, percentComplete float64) float64 {
	if percentComplete > 0.5 {
		// decrease lr
		return (1 - percentComplete) * (lrMax - lrMin) + lrMin
	}
	return percentComplete * (lrMax - lrMin) + lrMin
}