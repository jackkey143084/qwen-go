// Command qwen-openai — phase 5 (part 1): an OpenAI-compatible
// /v1/chat/completions passthrough server backed by the Go model.
//
// This is the Go twin of tiny-qwen's run.py `RemoteModel` client, inverted:
// instead of the agent talking *to* a remote OpenAI endpoint, this serves the
// OpenAI wire protocol locally so any OpenAI-compatible client (curl, python
// openai, other agents) can talk to the model directly.
//
//   GOGC=40 GOMEMLIMIT=2600MiB go run ./cmd/qwen-openai -model <dir> -addr :8080
//
// Supports non-streamed and streamed (SSE) completions, n=1, and the Qwen chat
// template. temperature/max_tokens are accepted for wire compatibility; the
// decoder is greedy and stops at <|im_end|>/<|im_start|>/<|endoftext|>.
package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"log"
	"net/http"
	"os"
	"strings"

	"tinyqwengo/model"
	"tinyqwengo/tokenizer"
)

const (
	sysTemplate   = "<|im_start|>system\n%s<|im_end|>\n"
	userTemplate  = "<|im_start|>user\n%s<|im_end|>\n"
	assistantOpen = "<|im_start|>assistant\n"
	thinkOpen     = " thinking\n"
	thinkClosed   = " thinking\n\n response\n\n"
	assistantWrap = "<|im_start|>assistant\n%s<|im_end|>\n"
	toolWrap      = "<|im_start|>tool\n%s<|im_end|>\n"
)

// --- OpenAI wire types -----------------------------------------------------

type chatReq struct {
	Model      string    `json:"model"`
	Messages   []message `json:"messages"`
	Stream     bool      `json:"stream"`
	MaxTokens  int       `json:"max_tokens"`
	MaxNew     *int      `json:"max_new_tokens"`
	Temperature float64  `json:"temperature"`
}

// content is a string or a list of text blocks ({type,text}). Unmarshal into
// raw and normalize to plain text — the Qwen template consumes text only.
func (r *chatReq) textOf(i int) string {
	var v any
	if err := json.Unmarshal(r.Messages[i].Content, &v); err != nil {
		return ""
	}
	switch t := v.(type) {
	case string:
		return t
	case []any:
		var b strings.Builder
		for _, block := range t {
			bm, ok := block.(map[string]any)
			if !ok {
				continue
			}
			if txt, ok := bm["text"].(string); ok {
				if b.Len() > 0 {
					b.WriteRune('\n')
				}
				b.WriteString(txt)
			}
			// image blocks are dropped text-wise (vision tower not ported)
		}
		return b.String()
	default:
		return ""
	}
}

type message struct {
	Role      string          `json:"role"`
	Content   json.RawMessage `json:"content"`
	ToolCallID string         `json:"tool_call_id"`
}

// ---------------------------------------------------------------------------

func stopTokens(t *tokenizer.Tokenizer) map[int]bool {
	stop := map[int]bool{}
	for _, name := range []string{"<|im_end|>", "<|im_start|>", "<|endoftext|>"} {
		if id, ok := t.ID(name); ok {
			stop[id] = true
		}
	}
	return stop
}

func renderPrompt(req *chatReq, system string, think bool) string {
	var b strings.Builder
	if system != "" {
		b.WriteString(fmt.Sprintf(sysTemplate, system))
	}
	for i, m := range req.Messages {
		text := req.textOf(i)
		switch m.Role {
		case "system":
			b.WriteString(fmt.Sprintf(sysTemplate, text))
		case "user":
			b.WriteString(fmt.Sprintf(userTemplate, text))
		case "assistant":
			if text != "" {
				b.WriteString(fmt.Sprintf(assistantWrap, text))
			}
		case "tool":
			b.WriteString(fmt.Sprintf(toolWrap, text))
		default:
			b.WriteString(fmt.Sprintf(userTemplate, text))
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

type server struct {
	m    *model.Model
	tok  *tokenizer.Tokenizer
	stop map[int]bool
}

func (s *server) chat(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}
	defer r.Body.Close()
	var req chatReq
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeErr(w, "request body is not valid JSON: "+err.Error())
		return
	}
	if len(req.Messages) == 0 {
		writeErr(w, "messages is required")
		return
	}

	maxNew := req.MaxTokens
	if req.MaxNew != nil {
		maxNew = *req.MaxNew
	}
	if maxNew <= 0 {
		maxNew = 256
	}

	system := "You are a helpful assistant."
	if len(req.Messages) > 0 && req.Messages[0].Role == "system" {
		system = req.textOf(0)
		req.Messages = req.Messages[1:]
	}

	prompt := renderPrompt(&req, system, true)
	ids := s.tok.EncodeWithSpecial(prompt)

	if req.Stream {
		s.stream(w, ids, maxNew)
		return
	}
	s.once(w, ids, maxNew)
}

func (s *server) once(w http.ResponseWriter, ids []int, maxNew int) {
	var sb strings.Builder
	if err := s.m.Generate(ids, maxNew, s.stop, func(id int) {
		sb.WriteString(s.tok.Decode([]int{id}))
	}); err != nil {
		writeErr(w, "generation failed: "+err.Error())
		return
	}
	text := strings.TrimSpace(sb.String())
	resp := map[string]any{
		"id":      "chatcmpl-local",
		"object":  "chat.completion",
		"created": 0,
		"model":   "qwen-go",
		"choices": []map[string]any{{
			"index":         0,
			"finish_reason": "stop",
			"message":       map[string]any{"role": "assistant", "content": text},
		}},
		"usage": map[string]any{
			"prompt_tokens":     len(ids),
			"completion_tokens": 0,
			"total_tokens":      len(ids),
		},
	}
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(resp)
}

func (s *server) stream(w http.ResponseWriter, ids []int, maxNew int) {
	fl, ok := w.(http.Flusher)
	if !ok {
		writeErr(w, "streaming unsupported by the HTTP server")
		return
	}
	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")
	w.Header().Set("X-Accel-Buffering", "no")

	writeChunk := func(payload map[string]any) {
		chunk, _ := json.Marshal(payload)
		fmt.Fprintf(w, "data: %s\n\n", chunk)
		fl.Flush()
	}
	// role chunk first, matching the OpenAI SSE shape
	writeChunk(map[string]any{
		"choices": []map[string]any{{"index": 0, "delta": map[string]string{"role": "assistant"}}},
	})
	completion := 0
	if err := s.m.Generate(ids, maxNew, s.stop, func(id int) {
		completion++
		piece := s.tok.Decode([]int{id})
		writeChunk(map[string]any{
			"choices": []map[string]any{{"index": 0, "delta": map[string]string{"content": piece}}},
		})
	}); err != nil {
		// terminate stream on error rather than hang the client
	}
	writeChunk(map[string]any{
		"choices": []map[string]any{{"index": 0, "delta": map[string]any{}, "finish_reason": "stop"}},
	})
	fmt.Fprint(w, "data: [DONE]\n\n")
	fl.Flush()
}

func writeErr(w http.ResponseWriter, msg string) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusBadRequest)
	json.NewEncoder(w).Encode(map[string]any{
		"error": map[string]string{"message": msg, "type": "invalid_request_error"},
	})
}

func main() {
	modelDir := flag.String("model", "", "model directory (config.json + *.safetensors + tokenizer.json)")
	addr := flag.String("addr", ":8080", "listen address")
	flag.Parse()
	if *modelDir == "" {
		fmt.Fprintln(os.Stderr, "usage: qwen-openai -model <dir> [-addr :8080]")
		os.Exit(1)
	}

	fmt.Fprintf(os.Stderr, "loading %s ...\n", *modelDir)
	m, tok, err := model.FromPretrained(*modelDir)
	if err != nil {
		log.Fatalf("load: %v", err)
	}
	s := &server{m: m, tok: tok, stop: stopTokens(tok)}
	http.HandleFunc("/v1/chat/completions", s.chat)
	http.HandleFunc("/healthz", func(w http.ResponseWriter, r *http.Request) { w.Write([]byte("ok\n")) })
	fmt.Fprintf(os.Stderr, "listening on %s (POST /v1/chat/completions)\n", *addr)
	log.Fatal(http.ListenAndServe(*addr, nil))
}
