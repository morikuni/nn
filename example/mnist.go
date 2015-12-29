package main

import (
	"fmt"
	"github.com/morikuni/nn"
	"github.com/morikuni/nn/mnist"
	"math/rand"
	"os"
	"time"
)

func toBin(v float64) int {
	if v >= 0.5 {
		return 1
	} else {
		return 0
	}
}

func toFlag(n uint8) [10]float64 {
	var v [10]float64
	v[n] = 1.0
	return v
}

func main() {
	const (
		ALPHA       = 0.5
		MAX_LOOP    = 60000
		HIDDEN_SIZE = 15
	)
	rand.Seed(time.Now().UnixNano())

	trainSc, err := mnist.Open(os.Args[1], os.Args[2])
	if err != nil {
		fmt.Errorf("Error: %v", err)
	}
	trainData := make([][]float64, 60000)
	trainExpect := make([][]float64, 60000)
	for i := 0; trainSc.Next(); i++ {
		image := trainSc.Image()
		trainData[i] = make([]float64, 28*28)
		for j, v := range image.Buffer {
			trainData[i][j] = float64(v) / 255
		}
		flag := toFlag(image.Label)
		trainExpect[i] = flag[:]
	}

	evalSc, err := mnist.Open(os.Args[3], os.Args[4])
	if err != nil {
		fmt.Errorf("Error: %v", err)
	}
	evalData := make([][]float64, 10000)
	evalExpect := make([][]float64, 10000)
	for i := 0; evalSc.Next(); i++ {
		image := evalSc.Image()
		evalData[i] = make([]float64, 28*28)
		for j, v := range image.Buffer {
			evalData[i][j] = float64(v) / 255
		}
		flag := toFlag(image.Label)
		evalExpect[i] = flag[:]
	}

	in := make([]*nn.Neuron, len(trainData[0]))
	for i := range in {
		in[i] = new(nn.Neuron)
	}

	hidden := make([]*nn.Neuron, HIDDEN_SIZE)
	for i := range hidden {
		hidden[i] = new(nn.Neuron)
	}

	out := make([]*nn.Neuron, len(trainExpect[0]))
	for i := range out {
		out[i] = new(nn.Neuron)
	}

	il := new(nn.Layer)
	il.Add(in...)
	hl := new(nn.Layer)
	hl.Add(hidden...)
	ol := new(nn.Layer)
	ol.Add(out...)

	nn.ConnectRandomWeight(il, hl, -0.1, 0.1)
	nn.ConnectRandomWeight(hl, ol, -0.1, 0.1)

	for _, h := range hidden {
		h.Activate()
	}
	for _, o := range out {
		o.Activate()
	}

	subs := make([]nn.Subscription, len(out))
	for i := range subs {
		subs[i] = out[i].Out.Subscribe()
	}

	//学習
	for di := range trainData {

		for i, x := range in {
			x.Out.Send(trainData[di][i])
		}

		r := make([]float64, len(subs))
		for si, s := range subs {
			r[si] = <-s.Result()
		}

		// 出力層の誤差
		eo := make([]nn.BackError, len(out))
		for oi := range out {
			eo[oi] = subs[oi].Error(trainExpect[di][oi] - r[oi])
		}

		eh := ol.BackProp(eo, ALPHA)
		hl.BackProp(eh, ALPHA)
	}

	//結果
	success := 0
	fail := 0
	for i := range evalData {
		for j, x := range in {
			x.Out.Send(evalData[i][j])
		}

		r := make([]float64, len(subs))
		for si, s := range subs {
			r[si] = <-s.Result()
		}

		maxi := -1
		maxv := -1.0
		for ri, v := range r {
			if maxv < v {
				maxv = v
				maxi = ri
			}
		}
		if toBin(evalExpect[i][maxi]) == toBin(1) {
			success++
		} else {
			fail++
		}
	}

	fmt.Println("success", success)
	fmt.Println("fail", fail)
}
