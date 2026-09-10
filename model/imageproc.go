package model

// Pure-Go image preprocessing, matching tiny_qwen/processor.py bit-for-bit:
// PIL-exact bicubic-antialias resize, (x/255-0.5)/0.5 normalize, duplicate
// the still frame to temporal_patch_size=2 frames, then patchify with the
// 2x2 spatial-merge ordering so the encoder can merge every consecutive
// 4 patches into one token.

import (
	"fmt"
	"image"
	_ "image/gif"
	_ "image/jpeg"
	_ "image/png"
	"math"
	"os"
)

// ImageGrid is the (t, h, w) patch grid of one processed image.
type ImageGrid struct {
	T, H, W int
}

// PixelsPerPatch is channels * temporal_patch_size * patch * patch.
const PixelsPerPatch = 3 * 2 * 16 * 16 // 1536

const (
	spatialPatchsize  = 16
	spatialMergeSize  = 2
	temporalPatchSize = 2
	minPixels         = 65536
	maxPixels         = 16777216
)

// ProcessImage loads an image file and returns patch vectors (N, 1536)
// plus the patch grid, mirroring Processor._process_image.
func ProcessImage(path string) ([]float32, ImageGrid, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, ImageGrid{}, err
	}
	defer f.Close()
	img, _, err := image.Decode(f)
	if err != nil {
		return nil, ImageGrid{}, fmt.Errorf("decode %s: %w", path, err)
	}

	b := img.Bounds()
	h0, w0 := b.Dy(), b.Dx()
	rh, rw := ResizeDims(h0, w0)
	resized := resizeBicubicPIL(img, rw, rh)

	// normalize to [-1, 1]: (x/255 - 0.5) / 0.5
	norm := make([]float32, rw*rh*3)
	for y := 0; y < rh; y++ {
		for x := 0; x < rw; x++ {
			r, g, bl, _ := resized.At(x, y).RGBA() // 16-bit
			norm[(y*rw+x)*3+0] = float32(r>>8)/127.5 - 1
			norm[(y*rw+x)*3+1] = float32(g>>8)/127.5 - 1
			norm[(y*rw+x)*3+2] = float32(bl>>8)/127.5 - 1
		}
	}

	gridH := rh / spatialPatchsize
	gridW := rw / spatialPatchsize
	out := make([]float32, gridH*gridW*PixelsPerPatch)

	// patchify with the merge ordering. Python does:
	//   reshape(t, tp, C, hB, ms, ps, wB, ms, ps).transpose(0,3,6,4,7,2,1,5,8)
	// with t=1: (hB, wB, ms_h, ms_w, C, tp, ps, ps). Since the two temporal
	// frames are identical, frame index is a no-op inside the dot product
	// only if we keep order (C, tp, py, px) per feature — we do.
	for hb := 0; hb < gridH/spatialMergeSize; hb++ {
		for wb := 0; wb < gridW/spatialMergeSize; wb++ {
			for mh := 0; mh < spatialMergeSize; mh++ {
				for mw := 0; mw < spatialMergeSize; mw++ {
					patch := (hb*(gridW/spatialMergeSize)+wb)*spatialMergeSize*spatialMergeSize +
						mh*spatialMergeSize + mw
					py0 := (hb*spatialMergeSize + mh) * spatialPatchsize
					px0 := (wb*spatialMergeSize + mw) * spatialPatchsize
					dst := out[patch*PixelsPerPatch : (patch+1)*PixelsPerPatch]
					for c := 0; c < 3; c++ {
						for py := 0; py < spatialPatchsize; py++ {
							src := norm[((py0+py)*rw+px0)*3:]
							for px := 0; px < spatialPatchsize; px++ {
								v := src[px*3+c]
								// feature layout (C, tp, py, px); tp=2 identical frames
								dst[(c*temporalPatchSize+0)*spatialPatchsize*spatialPatchsize+py*spatialPatchsize+px] = v
								dst[(c*temporalPatchSize+1)*spatialPatchsize*spatialPatchsize+py*spatialPatchsize+px] = v
							}
						}
					}
				}
			}
		}
	}
	return out, ImageGrid{T: 1, H: gridH, W: gridW}, nil
}

// ResizeDims mirrors Processor._resize_image: round h/w to multiples of 32,
// clamp the area between min_pixels and max_pixels.
func ResizeDims(height, width int) (int, int) {
	factor := spatialPatchsize * spatialMergeSize
	tBar := temporalPatchSize // ceil(num_frames/2)*2 for a single still
	hBar := roundInt(float64(height)/float64(factor)) * factor
	wBar := roundInt(float64(width)/float64(factor)) * factor
	if tBar*hBar*wBar > maxPixels {
		beta := math.Sqrt(float64(height*width) / float64(maxPixels))
		hBar = maxInt(factor, floorInt(float64(height)/beta/float64(factor))*factor)
		wBar = maxInt(factor, floorInt(float64(width)/beta/float64(factor))*factor)
	} else if hBar*wBar < minPixels {
		beta := math.Sqrt(float64(minPixels) / float64(height*width))
		hBar = ceilInt(float64(height)*beta/float64(factor)) * factor
		wBar = ceilInt(float64(width)*beta/float64(factor)) * factor
	}
	return hBar, wBar
}

func roundInt(v float64) int { return int(math.Floor(v + 0.5)) }
func ceilInt(v float64) int  { return int(math.Ceil(v)) }
func floorInt(v float64) int { return int(math.Floor(v)) }
func maxInt(a, b int) int {
	if a > b {
		return a
	}
	return b
}

func cubicKernel(x float64) float64 {
	x = math.Abs(x)
	if x < 1 {
		return 1.5*x*x*x - 2.5*x*x + 1
	} else if x < 2 {
		return -0.5*x*x*x + 2.5*x*x - 4*x + 2
	}
	return 0
}

// Pillow's 8-bit resample path: 22-bit fixed-point coefficients, int32
// accumulation with a rounding bias, clip to 0..255 after each pass.
const precisionBits = 32 - 8 - 2

// resizeBicubicPIL matches PIL Image.resize(size, BICUBIC) bit-for-bit:
// verbatim Resample.c window bounds (C trunc with +-0.5), dropped-tap
// renormalization at edges, uint8 intermediate between the two passes.
func resizeBicubicPIL(img image.Image, dw, dh int) *image.RGBA {
	b := img.Bounds()
	sw, sh := b.Dx(), b.Dy()
	src := make([]uint8, sw*sh*3)
	for y := 0; y < sh; y++ {
		for x := 0; x < sw; x++ {
			r, g, bl, _ := img.At(b.Min.X+x, b.Min.Y+y).RGBA()
			p := (y*sw + x) * 3
			src[p+0] = uint8(r >> 8)
			src[p+1] = uint8(g >> 8)
			src[p+2] = uint8(bl >> 8)
		}
	}
	horiz := resizeAxis8(src, sh, sw, dw, 3)
	return toRGBA(resizeVertical8(horiz, sh, dw, dh), dw, dh)
}

// resizeVertical8: src has inH rows of width*3 samples; resize the row
// count to outH (Pillow's second pass, uint8 intermediate).
func resizeVertical8(src []uint8, inH, width, outH int) []uint8 {
	scale := float64(inH) / float64(outH)
	filterscale := scale
	if filterscale < 1 {
		filterscale = 1
	}
	support := 2.0 * filterscale
	invFS := 1.0 / filterscale
	dst := make([]uint8, outH*width*3)
	for o := 0; o < outH; o++ {
		center := (float64(o) + 0.5) * scale
		xmin := int(center - support + 0.5)
		if xmin < 0 {
			xmin = 0
		}
		xmax := int(center + support + 0.5)
		if xmax > inH {
			xmax = inH
		}
		taps := xmax - xmin
		ki := make([]int32, taps)
		for x := 0; x < taps; x++ {
			w := cubicKernel((float64(x+xmin)-center+0.5)*invFS) * (1 << precisionBits)
			if w < 0 {
				ki[x] = int32(w - 0.5)
			} else {
				ki[x] = int32(w + 0.5)
			}
		}
		rnd := int32(1) << (precisionBits - 1)
		for r := 0; r < width; r++ {
			for c := 0; c < 3; c++ {
				var acc int32
				for x := 0; x < taps; x++ {
					acc += int32(src[(xmin+x)*width*3+r*3+c]) * ki[x]
				}
				acc = (acc + rnd) >> precisionBits
				if acc < 0 {
					acc = 0
				} else if acc > 255 {
					acc = 255
				}
				dst[o*width*3+r*3+c] = uint8(acc)
			}
		}
	}
	return dst
}

// resizeAxis8 resizes along the axis of length inLen (rows entries of
// inLen*ch samples); out has outLen*ch per row.
func resizeAxis8(src []uint8, rows, inLen, outLen, ch int) []uint8 {
	scale := float64(inLen) / float64(outLen)
	filterscale := scale
	if filterscale < 1 {
		filterscale = 1
	}
	support := 2.0 * filterscale
	invFS := 1.0 / filterscale
	dst := make([]uint8, rows*outLen*ch)
	for o := 0; o < outLen; o++ {
		center := (float64(o) + 0.5) * scale
		xmin := int(center - support + 0.5) // C trunc semantics
		if xmin < 0 {
			xmin = 0
		}
		xmax := int(center + support + 0.5)
		if xmax > inLen {
			xmax = inLen
		}
		taps := xmax - xmin
		ki := make([]int32, taps)
		sum := 0
		for x := 0; x < taps; x++ {
			w := cubicKernel((float64(x+xmin)-center+0.5)*invFS) * (1 << precisionBits)
			if w < 0 {
				ki[x] = int32(w - 0.5)
			} else {
				ki[x] = int32(w + 0.5)
			}
			sum += int(ki[x])
		}
		rnd := int32(1) << (precisionBits - 1)
		for r := 0; r < rows; r++ {
			for c := 0; c < ch; c++ {
				var acc int32
				for x := 0; x < taps; x++ {
					acc += int32(src[r*inLen*ch+(xmin+x)*ch+c]) * ki[x]
				}
				acc = (acc + rnd) >> precisionBits
				if acc < 0 {
					acc = 0
				} else if acc > 255 {
					acc = 255
				}
				dst[r*outLen*ch+o*ch+c] = uint8(acc)
			}
		}
	}
	return dst
}

func toRGBA(px []uint8, w, h int) *image.RGBA {
	out := image.NewRGBA(image.Rect(0, 0, w, h))
	copy(out.Pix, pixToRGBA(px, w, h))
	return out
}

func pixToRGBA(px []uint8, w, h int) []uint8 {
	buf := make([]uint8, w*h*4)
	for p := 0; p < w*h; p++ {
		buf[p*4+0] = px[p*3+0]
		buf[p*4+1] = px[p*3+1]
		buf[p*4+2] = px[p*3+2]
		buf[p*4+3] = 255
	}
	return buf
}

// DumpPixels writes the patch buffer as little-endian f32 (parity debugging).
func DumpPixels(pixels []float32, path string) {
	f, err := os.Create(path)
	if err != nil {
		panic(err)
	}
	defer f.Close()
	buf := make([]byte, 4*len(pixels))
	for i, v := range pixels {
		bits := math.Float32bits(v)
		buf[i*4] = byte(bits)
		buf[i*4+1] = byte(bits >> 8)
		buf[i*4+2] = byte(bits >> 16)
		buf[i*4+3] = byte(bits >> 24)
	}
	if _, err := f.Write(buf); err != nil {
		panic(err)
	}
}
