package main

import (
	"encoding/json"
	"strings"
	"testing"
)

func mustMsg(t *testing.T, raw string) message {
	t.Helper()
	var m message
	if err := json.Unmarshal([]byte(raw), &m); err != nil {
		t.Fatalf("unmarshal message: %v", err)
	}
	return m
}

func TestTextOfStringContent(t *testing.T) {
	req := chatReq{Messages: []message{mustMsg(t, `{"role":"user","content":"hi"}`)}}
	if got := req.textOf(0); got != "hi" {
		t.Fatalf("string content: got %q", got)
	}
}

func TestTextOfBlockContent(t *testing.T) {
	req := chatReq{Messages: []message{mustMsg(t,
		`{"role":"user","content":[{"type":"text","text":"one"},{"type":"image","image":"x.png"},{"type":"text","text":"two"}]}`)}}
	if got := req.textOf(0); got != "one\ntwo" {
		t.Fatalf("block content: got %q, want \"one\\ntwo\"", got)
	}
}

func TestRenderPromptStructure(t *testing.T) {
	req := chatReq{Messages: []message{
		mustMsg(t, `{"role":"user","content":"hello"}`),
		mustMsg(t, `{"role":"assistant","content":"hi"}`),
		mustMsg(t, `{"role":"tool","content":"out"}`),
	}}
	out := renderPrompt(&req, "be brief", true)
	for _, want := range []string{
		"<|im_start|>system\nbe brief<|im_end|>",
		"<|im_start|>user\nhello<|im_end|>",
		"<|im_start|>assistant\nhi<|im_end|>",
		"<|im_start|>tool\nout<|im_end|>",
		"<|im_start|>assistant\n",
		" thinking\n",
	} {
		if !strings.Contains(out, want) {
			t.Errorf("renderPrompt missing %q\nfull:\n%s", want, out)
		}
	}
	if count := strings.Count(out, "system\nbe brief<|im_end|>"); count != 1 {
		t.Errorf("system prompt rendered %d times, want exactly 1", count)
	}
}

// chat() lifts the leading system role out of the message list before
// renderPrompt, so a system message + a separate system arg must not double.
func TestChatDedupsLeadingSystem(t *testing.T) {
	req := &chatReq{Messages: []message{
		mustMsg(t, `{"role":"system","content":"you are terse"}`),
		mustMsg(t, `{"role":"user","content":"hi"}`),
	}}
	system := "you are terse"
	if req.Messages[0].Role == "system" {
		system = req.textOf(0)
		req.Messages = req.Messages[1:]
	}
	out := renderPrompt(req, system, false)
	if got := strings.Count(out, "system\nyou are terse<|im_end|>"); got != 1 {
		t.Errorf("leading system not deduped: rendered %d times", got)
	}
}
