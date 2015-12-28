package main

import (
	"fmt"
	"github.com/morikuni/nn"
	"math/rand"
	"time"
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
	rand.Seed(time.Now().UnixNano())

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

	in := make([]*nn.Neuron, len(data[0]))
	for i := range in {
		in[i] = new(nn.Neuron)
	}

	hidden := make([]*nn.Neuron, HIDDEN_SIZE)
	for i := range hidden {
		hidden[i] = new(nn.Neuron)
	}

	out := make([]*nn.Neuron, len(expect[0]))
	for i := range out {
		out[i] = new(nn.Neuron)
	}

	il := new(nn.Layer)
	il.Add(in...)
	hl := new(nn.Layer)
	hl.Add(hidden...)
	ol := new(nn.Layer)
	ol.Add(out...)

	il.ConnectRandomWeight(hl, -0.1, 0.1)
	hl.ConnectRandomWeight(ol, -0.1, 0.1)

	for _, h := range hidden {
		h.Activate()
	}
	for _, o := range out {
		o.Activate()
	}

	cr := make([]chan float64, len(out))
	for i := range cr {
		cr[i] = nn.Subscribe(out[i])
	}

	//学習
	for x := 0; x < MAX_LOOP; x++ {
		n := rand.Intn(len(data))

		for i, x := range in {
			x.Out.Send(data[n][i])
		}

		r := make([]float64, len(cr))
		for ci, c := range cr {
			r[ci] = <-c
		}

		// 出力層の誤差
		eo := make([]float64, len(out))
		for oi := range out {
			eo[oi] = (expect[n][oi] - r[oi]) * r[oi] * (1 - r[oi])
		}

		// 隠れ層の誤差
		eh := make([]float64, len(hidden))
		for hi, h := range hidden {
			for oi, o := range out {
				link, _ := h.FindLinkTo(o)
				eh[hi] += eo[oi] * link.Weight * link.Last() * (1 - link.Last())
			}
		}

		// 出力層の更新
		for oi, o := range out {
			o.Bias += ALPHA * eo[oi]
			for _, l := range o.In.Links {
				l.Weight += ALPHA * eo[oi] * l.Last()
			}
		}

		// 隠れ層の更新
		for hi, h := range hidden {
			h.Bias += ALPHA * eh[hi]
			for _, l := range h.In.Links {
				l.Weight += ALPHA * eh[hi] * l.Last()
			}
		}

		// 全ての入力について2乗誤差を足す
		se := 0.0
		for i := range data {
			for j, x := range in {
				x.Out.Send(data[i][j])
			}

			r := make([]float64, len(cr))
			for ci, c := range cr {
				r[ci] = <-c
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
			x.Out.Send(data[i][j])
		}

		r := make([]float64, len(cr))
		for ci, c := range cr {
			r[ci] = <-c
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
