package main

// vision-dump: parity harness for the vision tower. Prints checksums in the
// same "name sum sumsq" format as the Python reference dump
// (/tmp/vision_ref.py) so QWENGO_TRACE=1 output can be diffed line by line.

import (
	"encoding/json"
	"flag"
	"fmt"
	"os"

	"tinyqwengo/model"
)

func main() {
	dir := flag.String("model", "/opt/models/Qwen3.5-0.8B", "checkpoint dir")
	imagePath := flag.String("image", "", "image file")
	text := flag.String("text", "What is in this image?", "user text")
	maxNew := flag.Int("n", 8, "tokens to generate")
	flag.Parse()

	m, tok, err := model.FromPretrained(*dir)
	if err != nil {
		fmt.Fprintln(os.Stderr, "load:", err)
		os.Exit(1)
	}
	pixels, grid, err := model.ProcessImage(*imagePath)
	if err != nil {
		fmt.Fprintln(os.Stderr, "image:", err)
		os.Exit(1)
	}

	// replicate Processor.__call__ for a single user message with one image
	padCount := (grid.T * grid.H * grid.W) / 4
	prompt := "<|im_start|>user\n" + *text + "<|vision_start|>" +
		"<|image_pad|>"
	for i := 1; i < padCount; i++ {
		prompt += "<|image_pad|>"
	}
	prompt += "<|vision_end|><|im_end|>\n<|im_start|>assistant\n<think>\n"

	ids := tok.EncodeWithSpecial(prompt)
	pretty, _ := json.Marshal(map[string]any{
		"prompt_len": len(ids),
		"grid":       grid,
		"pixels_n":   len(pixels) / 1536,
	})
	fmt.Println(string(pretty))

	stop := map[int]bool{}
	for _, name := range []string{"<|im_end|>", "<|im_start|>", "<|endoftext|>"} {
		sids := tok.EncodeWithSpecial(name)
		if len(sids) == 1 {
			stop[sids[0]] = true
		}
	}
	if err := m.GenerateWithImage(ids, pixels, grid, *maxNew, stop, func(t int) {
		fmt.Println("tok", t)
	}); err != nil {
		fmt.Fprintln(os.Stderr, "generate:", err)
		os.Exit(1)
	}
}
