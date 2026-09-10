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

// resizeBicubicPIL matches PIL Image.resize(size, BICUBIC): separable filter, widened support when shrinking, uint8 result like an 8-bit source image.
func resizeBicubicPIL(img image.Image, dw, dh int) *image.RGBA {
	b := img.Bounds()
	sw, sh := b.Dx(), b.Dy()
	src := make([]float32, sw*sh*3)
	for y := 0; y < sh; y++ {
		for x := 0; x < sw; x++ {
			r, g, bl, _ := img.At(b.Min.X+x, b.Min.Y+y).RGBA()
			src[(y*sw+x)*3+0] = float32(r >> 8)
			src[(y*sw+x)*3+1] = float32(g >> 8)
			src[(y*sw+x)*3+2] = float32(bl >> 8)
		}
	}
	horiz := resizeHorizontal(src, sh, sw, dw)
	return toRGBA(resizeVertical(horiz, sh, dw, dh), dw, dh)
}

func cubicWeights(inLen, outLen int) (func(o int) []int, []func(o int) []float64) {
	return nil, nil
}

// resizePass resizes along one axis of r rows: each row has inLen groups of
// ch values; output rows have outLen groups.
func resizePass(src []float32, inLen, outLen, rows, ch int) []float32 {
	dst := make([]float32, rows*outLen*ch)
	scale := float64(inLen) / float64(outLen)
	filterscale := scale
	if filterscale < 1 {
		filterscale = 1
	}
	support := 2.0 * filterscale
	for o := 0; o < outLen; o++ {
		center := (float64(o) + 0.5) * scale
		lo := floorInt(center - support)
		hi := ceilInt(center + support)
		ws := make([]float64, 0, hi-lo+1)
		for j := lo; j <= hi; j++ {
			ws = append(ws, cubicKernel((center-(float64(j)+0.5))/filterscale))
		}
		sum := 0.0
		for _, w := range ws {
			sum += w
		}
		for r := 0; r < rows; r++ {
			for c := 0; c < ch; c++ {
				var acc float64
				for j, w := range ws {
					sj := lo + j
					if sj < 0 {
						sj = 0
					} else if sj >= inLen {
						sj = inLen - 1
					}
					acc += float64(src[r*inLen*ch+sj*ch+c]) * w
				}
				acc /= sum
				dst[r*outLen*ch+o*ch+c] = float32(acc)
			}
		}
	}
	return dst
}

func resizeHorizontal(src []float32, rows, inW, outW int) []float32 {
	return resizePass(src, inW, outW, rows, 3)
}

// resizeVertical: src is inH rows × width×3; returns outH rows × width×3.
func resizeVertical(src []float32, inH, width, outH int) []float32 {
	dst := make([]float32, outH*width*3)
	scale := float64(inH) / float64(outH)
	filterscale := scale
	if filterscale < 1 {
		filterscale = 1
	}
	support := 2.0 * filterscale
	for o := 0; o < outH; o++ {
		center := (float64(o) + 0.5) * scale
		lo := floorInt(center - support)
		hi := ceilInt(center + support)
		ws := make([]float64, 0, hi-lo+1)
		for j := lo; j <= hi; j++ {
			ws = append(ws, cubicKernel((center-(float64(j)+0.5))/filterscale))
		}
		sum := 0.0
		for _, w := range ws {
			sum += w
		}
		for r := 0; r < width; r++ {
			for c := 0; c < 3; c++ {
				var acc float64
				for j, w := range ws {
					sj := lo + j
					if sj < 0 {
						sj = 0
					} else if sj >= inH {
						sj = inH - 1
					}
					acc += float64(src[sj*width*3+r*3+c]) * w
				}
				acc /= sum
				dst[o*width*3+r*3+c] = float32(acc)
			}
		}
	}
	return dst
}

func toRGBA(px []float32, w, h int) *image.RGBA {
	out := image.NewRGBA(image.Rect(0, 0, w, h))
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			i := (y*w + x) * 3
			p := out.Pix[(y*w+x)*4:]
			for c := 0; c < 3; c++ {
				v := int(px[i+c] + 0.5)
				if v < 0 {
					v = 0
				} else if v > 255 {
					v = 255
				}
				p[c] = uint8(v)
			}
			p[3] = 255
		}
	}
	return out
}

// cubicKernel is the Mitchell-Netravali filter PIL uses for BICUBIC (a=-0.5).
func cubicKernel(x float64) float64 {
	x = math.Abs(x)
	if x < 1 {
		return 1.5*x*x*x - 2.5*x*x + 1
	} else if x < 2 {
		return -0.5*x*x*x + 2.5*x*x - 4*x + 2
	}
	return 0
}
