package nn

import (
	"math/rand"
	"sync"
)

// Group is a group of Neurons.
type Group interface {
	// Activate activate all Neurons of the Group.
	Activate()

	// Inputs return Neurons that are used as input of the Group.
	Inputs() []*Neuron

	// Inputs return Neurons that are used as output of the Group.
	Outputs() []*Neuron

	// BackProp update all Link weights of the Group with learning rate. Then return Errors of previous(input) Group.
	BackProp([]BackError, float64) []BackError
}

// BackError contain error of Output of the Group.
type BackError struct {
	link *Link
	err  float64
}

// Connect connect all Outputs of the group to all Inputs of other group.
func Connect(from, to Group) {
	for _, fn := range from.Outputs() {
		for _, tn := range to.Inputs() {
			tn.In.Register(&fn.Out)
		}
	}
}

// ConnectRandomWeight is same as Connect, but Link weights are randomized.
func ConnectRandomWeight(from, to Group, min, max float64) {
	for _, fn := range from.Outputs() {
		for _, tn := range to.Inputs() {
			l := tn.In.Register(&fn.Out)
			l.Weight = (max-min)*rand.Float64() + min
		}
	}
}

// Layer is a group of Neurons.
type Layer struct {
	m       sync.Mutex
	Neurons []*Neuron
}

// Add add Neuron to a the Layer.
func (layer *Layer) Add(ns ...*Neuron) {
	layer.m.Lock()
	layer.Neurons = append(layer.Neurons, ns...)
	layer.m.Unlock()
}

// Activate is a implementation of Group.
func (layer *Layer) Activate() {
	for _, n := range layer.Neurons {
		n.Activate()
	}
}

// Inputs is a implementation of Group.
func (layer *Layer) Inputs() []*Neuron {
	return layer.Neurons
}

// Outputs is a implementation of Group.
func (layer *Layer) Outputs() []*Neuron {
	return layer.Neurons
}

// BackProp is a implementation of Group.
func (layer *Layer) BackProp(errs []BackError, rate float64) []BackError {
	var bes []BackError
	for _, n := range layer.Neurons {
		en := 0.0
		for _, ol := range n.Out.Links {
			for _, err := range errs {
				if err.link == ol {
					en += err.err
				}
			}
		}
		bes = append(bes, onBackProp(n, en, rate)...)
	}
	return bes
}

func onBackProp(n *Neuron, err, rate float64) []BackError {
	bes := make([]BackError, len(n.In.Links))
	for i, l := range n.In.Links {
		bes[i] = BackError{
			l,
			err * l.Weight * l.Last() * (1 - l.Last()),
		}
		l.Weight += rate * err * l.Last()
	}
	n.Bias += rate * err
	return bes
}
