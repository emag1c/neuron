package neurons

type Group struct {
	Name    string
	Config  *Config
	Sess    *Session
	Neurons []*Neuron
}

//func NewGroup(Name, conf *Config, sess *Session, neurons int) *Group {
//
//}
