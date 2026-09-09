//go:build freebsd

package model

import (
	"syscall"
	"unsafe"
)

// madviseWillNeed: FreeBSD's Go syscall package doesn't wrap madvise, so
// issue it directly. MADV_WILLNEED = 3 on FreeBSD (mman.h).
func madviseWillNeed(data []byte) {
	if len(data) == 0 {
		return
	}
	syscall.Syscall(syscall.SYS_MADVISE,
		uintptr(unsafe.Pointer(&data[0])), uintptr(len(data)), 3)
}
