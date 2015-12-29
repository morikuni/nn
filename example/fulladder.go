package main

import (
	"fmt"
	"github.com/morikuni/nn"
	"math/rand"
)

func toBin(v float64) int {
	if v >= 0.5 {
		return 1
	} else {
		return 0
	}
}

func main() {
	const (
		ALPHA       = 0.5
		MAX_LOOP    = 100000
		EPS         = 0.2
		HIDDEN_SIZE = 2
	)
	rand.Seed(123)

	data := [][]float64{
		{0, 0, 0},
		{0, 0, 1},
		{0, 1, 0},
		{0, 1, 1},
		{1, 0, 0},
		{1, 0, 1},
		{1, 1, 0},
		{1, 1, 1},
	}
	expect := [][]float64{
		{0, 0},
		{0, 1},
		{0, 1},
		{1, 0},
		{0, 1},
		{1, 0},
		{1, 0},
		{1, 1},
	}

	il := nn.NewLayer(len(data[0]))
	hl := nn.NewLayer(HIDDEN_SIZE)
	ol := nn.NewLayer(len(expect[0]))

	in := make([]nn.Output, len(data[0]))
	for ii, i := range il.Inputs() {
		i.OnReceive(func(n *nn.Neuron, v float64) {
			n.Out.Send(v)
		})
		i.In.Register(&in[ii])
	}

	out := ol.Outputs()

	nn.ConnectRandomWeight(il, hl, -0.1, 0.1)
	nn.ConnectRandomWeight(hl, ol, -0.1, 0.1)

	il.Activate()
	hl.Activate()
	ol.Activate()

	subs := make([]nn.Subscription, len(out))
	for i := range subs {
		subs[i] = out[i].Out.Subscribe()
	}

	//学習
	for x := 0; x < MAX_LOOP; x++ {
		n := rand.Intn(len(data))

		for i, x := range in {
			x.Send(data[n][i])
		}

		r := make([]float64, len(subs))
		for si, s := range subs {
			r[si] = <-s.Result()
		}

		// 出力層の誤差
		eo := make([]nn.BackError, len(out))
		for oi := range out {
			eo[oi] = subs[oi].Error(expect[n][oi] - r[oi])
		}

		eh := ol.BackProp(eo, ALPHA)
		hl.BackProp(eh, ALPHA)

		// 全ての入力について2乗誤差を足す
		se := 0.0
		for i := range data {
			for j, x := range in {
				x.Send(data[i][j])
			}

			r := make([]float64, len(subs))
			for si, s := range subs {
				r[si] = <-s.Result()
			}
			for oi := range out {
				s := (expect[i][oi] - r[oi])
				se += s * s
			}
		}
		if se < EPS {
			fmt.Println("done when", x)
			break
		}
	}

	//結果
	for i := range data {
		for j, x := range in {
			x.Send(data[i][j])
		}

		r := make([]float64, len(subs))
		for si, s := range subs {
			r[si] = <-s.Result()
		}

		fmt.Print(" input( ")
		for _, v := range data[i] {
			fmt.Print(toBin(v), " ")
		}
		fmt.Println(")")
		fmt.Print("output( ")
		for _, v := range r {
			fmt.Print(toBin(v), " ")
		}
		fmt.Print(") ( ")
		for _, v := range r {
			fmt.Printf("%f ", v)
		}
		fmt.Println(")")
		fmt.Print("expect( ")
		for _, v := range expect[i] {
			fmt.Print(toBin(v), " ")
		}
		fmt.Println(")")
		result := "success"
		for ei, v := range expect[i] {
			if toBin(r[ei]) != toBin(v) {
				result = "fail"
				break
			}
		}
		fmt.Println(result)
		fmt.Println()
	}
}
