// Package nn is neural network emurator.
package nn

import (
	"math"
	"sync"
)

// Sigmoid is sigmoid function.
func Sigmoid(x float64) float64 {
	return 1.0 / (1.0 + math.Exp(-1*x))
}

// Neuron is a element of neural network.
type Neuron struct {
	Out                Output
	In                 Input
	Bias               float64
	ActivationFunction func(float64) float64
	receiveHandler     func(*Neuron, float64)
	m                  sync.Mutex
}

func defaultHandler(n *Neuron, f float64) {
	n.Out.Send(n.ActivationFunction(f + n.Bias))
}

// Activate activate the Neuron.
// Activated Neuron wait a input and handle sum of input.
func (neuron *Neuron) Activate() {
	if neuron.receiveHandler == nil {
		neuron.receiveHandler = defaultHandler
	}
	if neuron.ActivationFunction == nil {
		neuron.ActivationFunction = Sigmoid
	}
	go func() {
		for {
			f := <-neuron.In.sum()
			neuron.receiveHandler(neuron, f)
		}
	}()
}

// FindLinkTo find link to the Neuron.
func (neuron *Neuron) FindLinkTo(to *Neuron) (*Link, bool) {
	for _, flp := range neuron.Out.Links {
		for _, tlp := range to.In.Links {
			if flp == tlp {
				return flp, true
			}
		}
	}
	return nil, false
}

// OnReceive register a handler that receives a sum of input.
func (neuron *Neuron) OnReceive(receiveHandler func(*Neuron, float64)) {
	neuron.m.Lock()
	neuron.receiveHandler = receiveHandler
	neuron.m.Unlock()
}

// Subscribe make a channel that receive a output of the Neuron
func Subscribe(neuron *Neuron) <-chan float64 {
	l := neuron.Out.subscribe()
	return l.c
}
