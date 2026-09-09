//go:build freebsd || linux

package model

import (
	"fmt"
	"io"
	"os"
	"syscall"

	"tinyqwengo/safetensors"
)

// openShard memory-maps a safetensors shard instead of reading it into the
// heap. On FreeBSD this means:
//   - the kernel pages weights in from the unified buffer cache on demand,
//     so two processes running the same model share physical RAM;
//   - MADV_WILLNEED hints the VM prefetcher so prefill doesn't stall on
//     page-ins;
//   - no malloc pressure — the Go heap only holds the converted float32
//     tensors, not a second copy of the file bytes.
//
// The converted tensors are still owned copies; the mapping is dropped right
// after loading.
func openShard(path string) (*safetensors.File, io.Closer, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, nil, err
	}
	st, err := f.Stat()
	if err != nil {
		f.Close()
		return nil, nil, err
	}
	size := st.Size()
	if size == 0 {
		f.Close()
		return nil, nil, fmt.Errorf("%s: empty file", path)
	}
	data, err := syscall.Mmap(int(f.Fd()), 0, int(size), syscall.PROT_READ, syscall.MAP_SHARED)
	if err != nil {
		f.Close()
		return nil, nil, fmt.Errorf("mmap %s: %w", path, err)
	}
	// Prefetch hint: tell the kernel we'll walk the whole file. Advisory only.
	madviseWillNeed(data)
	sf, err := safetensors.Parse(data)
	if err != nil {
		syscall.Munmap(data)
		f.Close()
		return nil, nil, err
	}
	return sf, f, nil
}

var _ = os.FileMode(0)
