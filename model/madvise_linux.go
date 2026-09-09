//go:build linux

package model

import "syscall"

func madviseWillNeed(data []byte) {
	_ = syscall.Madvise(data, syscall.MADV_WILLNEED)
}
