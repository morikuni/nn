package mnist

import (
	"encoding/binary"
	"fmt"
	"os"
)

const (
	// ImageWidth is a pixel width of image.
	ImageWidth = 28

	// ImageHeight is a pixel height of image.
	ImageHeight = 28

	// TrainSize is a size of training dataset.
	TrainSize = 60000

	// EvalSize is a size of evaluation dataset.
	EvalSize = 10000
)

// Image is a handwritten image.
type Image struct {
	Label  uint8
	Buffer [ImageWidth * ImageHeight]byte
}

// At return a pixel value at (x, y).
func (image *Image) At(x, y uint) byte {
	return image.Buffer[ImageWidth*y+x]
}

// Print print Image to stdout.
func (image *Image) Print() {
	for y := uint(0); y < ImageHeight; y++ {
		for x := uint(0); x < ImageWidth; x++ {
			v := image.At(x, y)
			if v > 200 {
				fmt.Print("#")
			} else if v > 100 {
				fmt.Print("+")
			} else {
				fmt.Print(" ")
			}
		}
		fmt.Println()
	}
}

// Scanner is file handler of MNIST dataset.
type Scanner struct {
	image *os.File
	label *os.File
	next  *Image
}

// Next check if next data exists and load a Image to memory.
func (scanner *Scanner) Next() bool {
	image := new(Image)
	_, err := scanner.image.Read(image.Buffer[:])
	if err != nil {
		return false
	}
	err = binary.Read(scanner.label, binary.BigEndian, &image.Label)
	if err != nil {
		return false
	}
	scanner.next = image
	return true
}

// Image returns a Image loaded at last Next().
func (scanner *Scanner) Image() *Image {
	return scanner.next
}

func (scanner *Scanner) Close() {
	scanner.image.Close()
	scanner.label.Close()
}

// Open open
func Open(image, label string) (*Scanner, error) {
	imageFile, err := os.Open(image)
	if err != nil {
		return nil, err
	}
	labelFile, err := os.Open(label)
	if err != nil {
		return nil, err
	}

	// skip first 4 byte (header).
	for i := 0; i < 4; i++ {
		var v uint32
		binary.Read(imageFile, binary.BigEndian, &v)
	}

	// skip first 2 byte (header).
	for i := 0; i < 2; i++ {
		var v uint32
		binary.Read(labelFile, binary.BigEndian, &v)
	}

	sc := new(Scanner)
	sc.image = imageFile
	sc.label = labelFile
	return sc, nil
}
