package safetensors

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
	"io"
	"os"
)

// A Plan is one tensor scheduled for a streaming write: the caller promises
// bufLen raw bytes, supplied later via Streamer.Write.
type Plan struct {
	Name  string
	Shape []int
	Dtype Dtype
	// Len is the raw byte length of the payload that will be written.
	Len int
}

// Writer streams tensors to a safetensors file without holding all payloads in
// memory at once: Open writes the header, then the caller feeds each planned
// tensor's bytes in plan order with Write. This keeps peak memory to one tensor
// while quantizing a multi-GB checkpoint.
type Writer struct {
	w   io.WriteSeeker
	buf []byte // scratch for padding
}

// CreateStreaming opens outFile and writes the header for the given plans. The
// caller must then call Write once per plan, in order.
func CreateStreaming(outFile string, plans []Plan) (*Writer, error) {
	type headerEntry struct {
		Dtype      Dtype    `json:"dtype"`
		Shape      []int    `json:"shape"`
		DataOffset [2]int64 `json:"data_offsets"`
	}
	header := map[string]headerEntry{}
	var cursor int64
	for _, p := range plans {
		if p.Dtype == "" || p.Len < 0 {
			return nil, fmt.Errorf("safetensors: bad plan %q", p.Name)
		}
		pad := (8 - cursor%8) % 8
		cursor += pad + int64(p.Len)
		header[p.Name] = headerEntry{Dtype: p.Dtype, Shape: p.Shape, DataOffset: [2]int64{cursor - int64(p.Len), cursor}}
	}
	hdrJSON, err := json.Marshal(header)
	if err != nil {
		return nil, err
	}
	for len(hdrJSON)%8 != 0 {
		hdrJSON = append(hdrJSON, ' ')
	}
	f, err := os.Create(outFile)
	if err != nil {
		return nil, err
	}
	var lenbuf [8]byte
	binary.LittleEndian.PutUint64(lenbuf[:], uint64(len(hdrJSON)))
	if _, err := f.Write(lenbuf[:]); err != nil {
		f.Close()
		return nil, err
	}
	if _, err := f.Write(hdrJSON); err != nil {
		f.Close()
		return nil, err
	}
	return &Writer{w: f, buf: make([]byte, 8)}, nil
}

// WriteAll writes a full plan-sized payload in one call (convenience for
// callers that hold the whole tensor). Trailing padding to the next 8-byte
// boundary is emitted automatically.
func (sw *Writer) WriteAll(payload []byte) error {
	n := len(payload)
	if _, err := sw.w.Write(payload); err != nil {
		return err
	}
	pad := (8 - n%8) % 8
	if pad > 0 {
		if _, err := sw.w.Write(sw.buf[:pad]); err != nil {
			return err
		}
	}
	return nil
}

// Close flushes and closes the underlying file. Padding to the final 8-byte
// boundary, if any payload is still pending, is not this API's concern (callers
// use WriteAll). Call Close once at the end.
func (sw *Writer) Close() error {
	if f, ok := sw.w.(*os.File); ok {
		return f.Close()
	}
	if c, ok := sw.w.(io.Closer); ok {
		return c.Close()
	}
	return nil
}

// Tensor is a fully-materialized tensor for the in-memory Write helper below.
type Tensor struct {
	Name  string
	Raw   []byte // raw bytes in the target dtype
	Shape []int
	Dtype Dtype
}

// Write serializes tensors to a safetensors file at path in one shot (the
// non-streaming helper; for large checkpoints prefer CreateStreaming).
func Write(path string, tensors []Tensor) error {
	plans := make([]Plan, len(tensors))
	for i, t := range tensors {
		if t.Dtype == "" || t.Dtype == BOOL {
			return fmt.Errorf("safetensors: invalid dtype %q for %q", t.Dtype, t.Name)
		}
		plans[i] = Plan{Name: t.Name, Shape: t.Shape, Dtype: t.Dtype, Len: len(t.Raw)}
	}
	sw, err := CreateStreaming(path, plans)
	if err != nil {
		return err
	}
	for _, t := range tensors {
		if err := sw.WriteAll(t.Raw); err != nil {
			sw.Close()
			return err
		}
	}
	return sw.Close()
}
