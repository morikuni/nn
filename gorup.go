package nn

import (
	"math/rand"
	"sync"
)

// Group is a group of Neurons.
type Group interface {
	// Connect connect all Outputs of the group to all Inputs of other group.
	Connect(Group)

	// ConnectRandomWeight is same as Connect, but Link weights area randomized.
	ConnectRandomWeight(Group, float64, float64)

	// Activate activate all Neurons of the group.
	Activate()

	// Inputs return Neurons that are used as input of the group.
	Inputs() []*Neuron
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

// Connect is a implementation of Group.
func (layer *Layer) Connect(to Group) {
	for _, fn := range layer.Neurons {
		for _, tn := range to.Inputs() {
			tn.In.Connect(&fn.Out)
		}
	}
}

// ConnectRandomWeight is a implementation of Group.
func (layer *Layer) ConnectRandomWeight(to Group, min, max float64) {
	for _, fn := range layer.Neurons {
		for _, tn := range to.Inputs() {
			l := tn.In.Connect(&fn.Out)
			l.Weight = (max-min)*rand.Float64() + min
		}
	}
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
