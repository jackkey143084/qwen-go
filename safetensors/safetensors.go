// Package safetensors reads Hugging Face safetensors files with zero
// external dependencies.
//
// A safetensors file is: 8-byte little-endian header length N, an N-byte
// JSON header describing each tensor (dtype, shape, byte offsets), then a
// raw data blob. We read the whole file into memory once and hand out
// []float32 views (converting on access) plus raw bytes for quantized
// tensors.
package safetensors

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"os"
	"unsafe"
)

// Dtype mirrors the dtypes we can meet in Qwen checkpoints.
type Dtype string

const (
	F32   Dtype = "F32"
	F16   Dtype = "F16"
	BF16  Dtype = "BF16"
	I64   Dtype = "I64"
	I32   Dtype = "I32"
	I8    Dtype = "I8"
	U8    Dtype = "U8"
	F64   Dtype = "F64"
	BOOL  Dtype = "BOOL"
	F8E4M3 Dtype = "F8_E4M3"
)

type TensorInfo struct {
	Dtype      Dtype    `json:"dtype"`
	Shape      []int    `json:"shape"`
	DataOffset [2]int64 `json:"data_offsets"`
}

// File is one loaded safetensors file.
type File struct {
	data   []byte
	Header map[string]TensorInfo `json:"-"`
}

// Open reads and parses a safetensors file.
func Open(path string) (*File, error) {
	raw, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	return Parse(raw)
}

// Parse parses an in-memory safetensors blob.
func Parse(raw []byte) (*File, error) {
	if len(raw) < 8 {
		return nil, fmt.Errorf("safetensors: file too small (%d bytes)", len(raw))
	}
	n := binary.LittleEndian.Uint64(raw[:8])
	if uint64(len(raw)) < 8+n {
		return nil, fmt.Errorf("safetensors: header length %d exceeds file", n)
	}
	headerBytes := raw[8 : 8+n]
	var header map[string]TensorInfo
	if err := json.Unmarshal(headerBytes, &header); err != nil {
		return nil, fmt.Errorf("safetensors: bad header json: %w", err)
	}
	delete(header, "__metadata__")
	return &File{data: raw, Header: header}, nil
}

// Keys lists tensor names, sorted for deterministic loading.
func (f *File) Keys() []string {
	keys := make([]string, 0, len(f.Header))
	for k := range f.Header {
		keys = append(keys, k)
	}
	// insertion order doesn't matter; sort for determinism
	for i := 1; i < len(keys); i++ {
		for j := i; j > 0 && keys[j] < keys[j-1]; j-- {
			keys[j], keys[j-1] = keys[j-1], keys[j]
		}
	}
	return keys
}

// Info returns the tensor descriptor.
func (f *File) Info(name string) (TensorInfo, bool) {
	t, ok := f.Header[name]
	return t, ok
}

// Raw returns the raw underlying bytes of a tensor (no conversion).
func (f *File) Raw(name string) ([]byte, []int, Dtype, error) {
	info, ok := f.Header[name]
	if !ok {
		return nil, nil, "", fmt.Errorf("safetensors: no tensor %q", name)
	}
	lo, hi := info.DataOffset[0], info.DataOffset[1]
	if hi > int64(len(f.data)) {
		return nil, nil, "", fmt.Errorf("safetensors: tensor %q out of range", name)
	}
	return f.data[lo:hi], info.Shape, info.Dtype, nil
}

// F32 returns the tensor converted to float32. Conversion happens on every
// call, so cache the result if you use it repeatedly.
func (f *File) F32(name string) ([]float32, []int, error) {
	raw, shape, dt, err := f.Raw(name)
	if err != nil {
		return nil, nil, err
	}
	out, err := toF32(raw, dt)
	if err != nil {
		return nil, nil, fmt.Errorf("safetensors: %q: %w", name, err)
	}
	return out, shape, nil
}

func toF32(raw []byte, dt Dtype) ([]float32, error) {
	switch dt {
	case F32:
		if len(raw)%4 != 0 {
			return nil, fmt.Errorf("f32 slice %d bytes not aligned", len(raw))
		}
		return rawToF32(raw), nil
	case F64:
		n := len(raw) / 8
		out := make([]float32, n)
		for i := 0; i < n; i++ {
			out[i] = float32(math.Float64frombits(binary.LittleEndian.Uint64(raw[i*8:])))
		}
		return out, nil
	case F16:
		n := len(raw) / 2
		out := make([]float32, n)
		for i := 0; i < n; i++ {
			out[i] = f16to32(binary.LittleEndian.Uint16(raw[i*2:]))
		}
		return out, nil
	case BF16:
		// bf16 is the top 16 bits of an f32: widen in place.
		n := len(raw) / 2
		out := make([]float32, n)
		for i := 0; i < n; i++ {
			out[i] = math.Float32frombits(uint32(binary.LittleEndian.Uint16(raw[i*2:])) << 16)
		}
		return out, nil
	case I8:
		out := make([]float32, len(raw))
		for i, b := range raw {
			out[i] = float32(int8(b))
		}
		return out, nil
	case U8:
		out := make([]float32, len(raw))
		for i, b := range raw {
			out[i] = float32(b)
		}
		return out, nil
	default:
		return nil, fmt.Errorf("unsupported dtype %s", dt)
	}
}

// rawToF32 converts little-endian f32 bytes to a float32 slice.
func rawToF32(b []byte) []float32 {
	n := len(b) / 4
	out := make([]float32, n)
	for i := 0; i < n; i++ {
		out[i] = math.Float32frombits(binary.LittleEndian.Uint32(b[i*4:]))
	}
	return out
}

func f16to32(h uint16) float32 {
	sign := uint32(h>>15) & 1
	exp := uint32(h>>10) & 0x1F
	frac := uint32(h) & 0x3FF
	var bits uint32
	switch {
	case exp == 0:
		if frac == 0 {
			bits = sign << 31
		} else {
			// subnormal
			e := uint32(127 - 15 + 1)
			for frac&0x400 == 0 {
				frac <<= 1
				e--
			}
			frac &= 0x3FF
			bits = sign<<31 | e<<23 | frac<<13
		}
	case exp == 0x1F:
		bits = sign<<31 | 0xFF<<23 | frac<<13
	default:
		bits = sign<<31 | (exp-15+127)<<23 | frac<<13
	}
	return math.Float32frombits(bits)
}

var _ = io.Discard // keep io imported for future mmap-style readers
