// Package nn is neural network emurator.
package nn

import (
	"math"
	"math/rand"
	"sync"
)

// Sigmoid is sigmoid function.
func Sigmoid(x float64) float64 {
	return 1.0 / (1.0 + math.Exp(-1*x))
}

// Link is connection of Neurons.
type Link struct {
	c      chan float64
	Weight float64
	last   float64
}

func (l *Link) receive() float64 {
	v := <-l.c
	l.last = v
	return v * l.Weight
}

func (l *Link) send(v float64) {
	l.c <- v
}

// Last return last value of this Link.
func (l *Link) Last() float64 {
	return l.last
}

type adapter struct {
	m     sync.Mutex
	Links []*Link
}

// Output is Neuron's output.
type Output adapter

// Send send a value to all receivers.
func (o *Output) Send(v float64) {
	for _, l := range o.Links {
		l.send(v)
	}
}

func (o *Output) subscribe() *Link {
	c := make(chan float64, 1)
	l := &Link{c, 1, math.MaxFloat64}
	o.m.Lock()
	o.Links = append(o.Links, l)
	o.m.Unlock()
	return l
}

// Input is Neuron's input.
type Input adapter

// Connect connect Output as a input.
func (i *Input) Connect(o *Output) *Link {
	l := o.subscribe()
	i.m.Lock()
	i.Links = append(i.Links, l)
	i.m.Unlock()
	return l
}

func (i *Input) sum() chan float64 {
	c := make(chan float64, 1)
	go func() {
		sum := 0.0
		for _, l := range i.Links {
			sum += l.receive()
		}
		c <- sum
	}()
	return c
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

// Activate activate a Neuron.
// Activated Neuron wait a input and handle sum of input.
func (n *Neuron) Activate() {
	if n.receiveHandler == nil {
		n.receiveHandler = defaultHandler
	}
	if n.ActivationFunction == nil {
		n.ActivationFunction = Sigmoid
	}
	go func() {
		for {
			f := <-n.In.sum()
			n.receiveHandler(n, f)
		}
	}()
}

// FindLinkTo find link to a Neuron.
func (n *Neuron) FindLinkTo(to *Neuron) (*Link, bool) {
	for _, flp := range n.Out.Links {
		for _, tlp := range to.In.Links {
			if flp == tlp {
				return flp, true
			}
		}
	}
	return nil, false
}

// OnReceive register a handler that receives a sum of input.
func (n *Neuron) OnReceive(receiveHandler func(*Neuron, float64)) {
	n.m.Lock()
	n.receiveHandler = receiveHandler
	n.m.Unlock()
}

// Layer is group of Neurons
type Layer struct {
	m       sync.Mutex
	Neurons []*Neuron
}

// Connect connect all Output(s) to target Layer's Input(s).
func (l *Layer) Connect(to *Layer) {
	for _, fn := range l.Neurons {
		for _, tn := range to.Neurons {
			tn.In.Connect(&fn.Out)
		}
	}
}

// ConnectRandomWeight connect all Output(s) to target Layer's Input(s) with random weight.
func (l *Layer) ConnectRandomWeight(to *Layer, min, max float64) {
	for _, fn := range l.Neurons {
		for _, tn := range to.Neurons {
			l := tn.In.Connect(&fn.Out)
			l.Weight = (max-min)*rand.Float64() + min
		}
	}
}

// Add add Neuron to a this Layer.
func (l *Layer) Add(ns ...*Neuron) {
	l.m.Lock()
	l.Neurons = append(l.Neurons, ns...)
	l.m.Unlock()
}

// Activate activate all Neurons in this Layer.
func (l *Layer) Activate() {
	for _, n := range l.Neurons {
		n.Activate()
	}
}

// Subscribe make channel that receive a output of a Neuron
func Subscribe(n *Neuron) chan float64 {
	l := n.Out.subscribe()

	return l.c
}
