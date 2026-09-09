package safetensors

import (
	"encoding/binary"
	"math"
	"path/filepath"
	"testing"
)

// TestWriteReadRoundTrip validates the streaming writer: written file registers,
// then re-parses via the reader with identical shapes/dtypes/data.
func TestWriteReadRoundTrip(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "model.safetensors")

	// one BF16 passthrough tensor and one F32 tensor (the quantized scale shape)
	bf16 := make([]byte, 8)
	for i := 0; i < 4; i++ {
		binary.LittleEndian.PutUint16(bf16[i*2:], 0x3F80) // 1.0 in bf16
	}
	f32 := make([]float32, 4)
	for i := range f32 {
		f32[i] = float32(i) * 1.5
	}

	plans := []Plan{
		{Name: "layers.0.self_attn.q_proj.weight", Shape: []int{4, 1}, Dtype: BF16, Len: len(bf16)},
		{Name: "layers.0.self_attn.q_proj.scale", Shape: []int{4, 1}, Dtype: F32, Len: len(f32) * 4},
	}
	sw, err := CreateStreaming(path, plans)
	if err != nil {
		t.Fatalf("CreateStreaming: %v", err)
	}
	if err := sw.WriteAll(bf16); err != nil {
		t.Fatalf("write bf16: %v", err)
	}
	scaleBytes := make([]byte, len(f32)*4)
	for i, v := range f32 {
		binary.LittleEndian.PutUint32(scaleBytes[i*4:], math.Float32bits(v))
	}
	if err := sw.WriteAll(scaleBytes); err != nil {
		t.Fatalf("write scale: %v", err)
	}
	if err := sw.Close(); err != nil {
		t.Fatalf("close: %v", err)
	}

	f, err := Open(path)
	if err != nil {
		t.Fatalf("re-Open: %v", err)
	}
	if len(f.Keys()) != 2 {
		t.Fatalf("want 2 keys, got %d", len(f.Keys()))
	}
	// validate the f32 scale round-trips
	vals, shape, err := f.F32("layers.0.self_attn.q_proj.scale")
	if err != nil {
		t.Fatalf("F32: %v", err)
	}
	if shape[0] != 4 || shape[1] != 1 {
		t.Fatalf("shape: %v", shape)
	}
	for i, v := range vals {
		if math.Abs(float64(v)-float64(f32[i])) > 1e-6 {
			t.Fatalf("scale[%d]=%v want %v", i, v, f32[i])
		}
	}
}

// TestWriteEmpty ensures an empty file still parses (header-only body).
func TestWriteEmpty(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "empty.safetensors")
	if err := Write(path, nil); err != nil {
		t.Fatalf("empty write: %v", err)
	}
	f, err := Open(path)
	if err != nil {
		t.Fatalf("open empty: %v", err)
	}
	if len(f.Keys()) != 0 {
		t.Fatalf("want 0 keys, got %d", len(f.Keys()))
	}
}
