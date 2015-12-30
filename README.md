# nn
nn helps creating Neural Network.

All neurons running on goroutine.

## Example

First neuron receive a value `1` then return a result by activation function.
Second neuron receive a result of first neuron then return a result.

```go
package main

import (
	"fmt"
	"github.com/morikuni/nn"
)

func main() {
	first := new(nn.Neuron)
	second := new(nn.Neuron)

	second.In.Register(&first.Out)

	pub := first.In.Publish()
	sub := second.Out.Subscribe()

	first.Activate()
	second.Activate()

	go func() {
		pub.Send(1)
	}()

	fmt.Println(<-sub.Result() == nn.Sigmoid(nn.Sigmoid(1)))
	// true
}
```

Change a weight of second neuron's input.

```go
...
  second.In.Links[0].Weight = 2.0
...
  fmt.Println(<-sub.Result() == nn.Sigmoid(2*nn.Sigmoid(1)))
  //true
...
```

Change first neuron's activation function.

```go
...
  first.ActivationFunction = func(v float64) float64 { return v }
  second.In.Links[0].Weight = 2.0
...
  fmt.Println(<-sub.Result() == nn.Sigmoid(2*1))
  //true
...
```


Another example using layer.
Create 3 neurons input layer and 2 neurons output layer.
```go
package main

import (
	"fmt"
	"math/rand"
	"github.com/morikuni/nn"
)

func main() {
	inputLayer := nn.NewPublisher(3)
	layer := nn.NewLayer(2)

	rand.Seed(123)
	nn.ConnectRandomWeight(inputLayer, layer, -1, 1)

	pubs := inputLayer.Publications()

	subs := make([]nn.Subscription, len(layer.Outputs()))
	for i, o := range layer.Outputs() {
		subs[i] = o.Subscribe()
	}

	layer.Activate()

	go func() {
		for i, p := range pubs {
			p.Send(float64(i))
		}
	}()

	for _, s := range subs {
		fmt.Println(<-s.Result())
	}
}

// 0.5069170823566078
// 0.6508370683765821
```

## mnist

Submodule mnist loads MNIST handwritten digit data.

```go
package main

import (
	"fmt"
	"github.com/morikuni/nn/mnist"
)

func main() {
	sc, err := mnist.Open("/path/to/train-images-idx3-ubyte", "/path/to/train-labels-idx1-ubyte")

	if err != nil {
		fmt.Errorf("%v", err)
	}

	if sc.Next() {
		image := sc.Image()
		fmt.Println("label", image.Label)
		fmt.Println("value at (x = 15, y = 15) = ", image.At(15, 15))
		image.Print()
	}
}
```

output

```
label 5
value at (x = 15, y = 15) = 186





                +++ +##+
           ++######+##+
        ##########
        ######++##
         ++###   +
           +#
           +#+
            +#
             ##++
              ###+
               +##+
                 ##+
                 ###
               ++###
             +#####+
           +######
          #####+
       +#####+
     +######+
    +####++




```
