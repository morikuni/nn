package nn

import (
	"math/rand"
	"sync"
)

// Group is a group of Neurons.
type Group interface {
	// Activate activate all Neurons of the group.
	Activate()

	// Inputs return Neurons that are used as input of the group.
	Inputs() []*Neuron

	// Inputs return Neurons that are used as output of the group.
	Outputs() []*Neuron
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
