//go:build !(freebsd || linux)

package model

import (
	"fmt"
	"io"
	"os"

	"tinyqwengo/safetensors"
)

// openShard falls back to a plain read on OSes without a usable syscall.Mmap
// in the stdlib surface (windows, darwin handled separately if ever needed).
func openShard(path string) (*safetensors.File, io.Closer, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	st, err := f.Stat()
	if err != nil {
		f.Close()
		return nil, err
	}
	data := make([]byte, st.Size())
	if _, err := f.ReadAt(data, 0); err != nil {
		f.Close()
		return nil, fmt.Errorf("read %s: %w", path, err)
	}
	sf, err := safetensors.Parse(data)
	if err != nil {
		syscall.Munmap(data)
		f.Close()
		return nil, nil, err
	}
	return sf, f, nil
}

var _ = os.FileMode(0)
