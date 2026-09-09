// cmd/tiny-qwen — streaming chat harness, the Go twin of tiny-qwen's run.py
// local mode. Greedy decoding, Qwen chat template, thinking toggle.
package main

import (
	"bufio"
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"strings"

	"tinyqwengo/model"
	"tinyqwengo/tokenizer"
)

const (
	sysTemplate   = "<|im_start|>system\n%s<|im_end|>\n"
	userTemplate  = "<|im_start|>user\n%s<|im_end|>\n"
	assistantOpen = "<|im_start|>assistant\n"
	thinkOpen     = "<think>\n"
	thinkClosed   = "<think>\n\n</think>\n\n"
)

func stopTokens(t *tokenizer.Tokenizer) map[int]bool {
	stop := map[int]bool{}
	for _, name := range []string{"<|im_end|>", "<|im_start|>", "<|endoftext|>"} {
		if id, ok := t.ID(name); ok {
			stop[id] = true
		}
	}
	return stop
}

func renderPrompt(history []string, system string, think bool) string {
	var b strings.Builder
	if system != "" {
		b.WriteString(fmt.Sprintf(sysTemplate, system))
	}
	// history alternates user / assistant starting at index 0
	for i, turn := range history {
		if i%2 == 0 {
			b.WriteString(fmt.Sprintf(userTemplate, turn))
		} else {
			b.WriteString(fmt.Sprintf(assistantOpen+"%s<|im_end|>\n", turn))
		}
	}
	b.WriteString(assistantOpen)
	if think {
		b.WriteString(thinkOpen)
	} else {
		b.WriteString(thinkClosed)
	}
	return b.String()
}

func main() {
	modelDir := flag.String("model", "", "model directory (config.json + *.safetensors + tokenizer.json)")
	maxNew := flag.Int("max-new-tokens", 256, "max tokens per reply")
	system := flag.String("system", "You are a helpful assistant.", "system prompt")
	think := flag.Bool("think", true, "enable the <think> block in the generation prompt")
	oneshot := flag.String("prompt", "", "one-shot mode: generate from this raw prompt and print token ids")
	flag.Parse()

	if *modelDir == "" {
		fmt.Fprintln(os.Stderr, "usage: tiny-qwen -model <dir> [--prompt <raw text> | -max-new-tokens N] ...")
		os.Exit(1)
	}

	fmt.Fprintf(os.Stderr, "loading %s ...\n", *modelDir)
	m, tok, err := model.FromPretrained(*modelDir)
	if err != nil {
		fmt.Fprintln(os.Stderr, "load:", err)
		os.Exit(1)
	}
	stop := stopTokens(tok)

	if *oneshot != "" {
		ids := tok.EncodeWithSpecial(*oneshot)
		fmt.Fprintf(os.Stderr, "prompt tokens (%d): %v\n", len(ids), ids)
		var got []int
		err := m.Generate(ids, *maxNew, map[int]bool{}, func(id int) {
			got = append(got, id)
		})
		if err != nil {
			fmt.Fprintln(os.Stderr, "generate:", err)
			os.Exit(1)
		}
		out, _ := json.Marshal(map[string]interface{}{"prompt": ids, "generated": got})
		fmt.Println(string(out))
		return
	}

	sc := bufio.NewScanner(os.Stdout)
	_ = sc
	in := bufio.NewScanner(os.Stdin)
	in.Buffer(make([]byte, 1<<20), 1<<20)
	var history []string
	fmt.Fprintf(os.Stderr, "ready. type a message; Ctrl-D to quit.\n")
	for {
		fmt.Print("> ")
		if !in.Scan() {
			break
		}
		line := strings.TrimSpace(in.Text())
		if line == "" {
			continue
		}
		history = append(history, line)
		prompt := renderPrompt(history, *system, *think)
		ids := tok.EncodeWithSpecial(prompt)
		fmt.Println()
		var sb strings.Builder
		err := m.Generate(ids, *maxNew, stop, func(id int) {
			piece := tok.Decode([]int{id})
			sb.WriteString(piece)
			fmt.Print(piece)
		})
		fmt.Println()
		if err != nil {
			fmt.Fprintln(os.Stderr, "generate:", err)
			os.Exit(1)
		}
		history = append(history, sb.String())
	}
}
