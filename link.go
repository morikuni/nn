package nn

import (
	"math"
	"sync"
)

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

// Subscribe make a Subscription.
func (o *Output) Subscribe() Subscription {
	return Subscription{
		o.subscribe(),
	}
}

// Subscription is output receiver. and create BackError.
type Subscription struct {
	l *Link
}

// Result make a channel that receive a output.
func (s *Subscription) Result() <-chan float64 {
	return s.l.c
}

// Error create a BackError that notify error to Group.
func (s *Subscription) Error(e float64) BackError {
	return BackError{
		s.l,
		e,
	}
}

// Publication is used to send a value.
type Publication struct {
	o *Output
}

// Send send a value to all connected Links.
func (p *Publication) Send(v float64) {
	p.o.Send(v)
}

// Input is Neuron's input.
type Input adapter

// Register connect a Output as a input.
func (i *Input) Register(o *Output) *Link {
	l := o.subscribe()
	i.m.Lock()
	i.Links = append(i.Links, l)
	i.m.Unlock()
	return l
}

func (i *Input) sum() <-chan float64 {
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

// Publish create a Publication.
func (i *Input) Publish() Publication {
	o := new(Output)
	i.Register(o)
	return Publication{o}
}
