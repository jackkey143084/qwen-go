// Package tokenizer is a minimal byte-level BPE tokenizer that reads a
// Hugging Face tokenizer.json (the format Qwen ships) — no external deps.
//
// Ported from tiny-qwen's Python (which uses the `tokenizers` lib): same
// pre-tokenizer rules, same byte↔unicode remap, same merge priority.
package tokenizer

import (
	"bufio"
	"encoding/json"
	"fmt"
	"os"
	"sort"
	"strings"
	"unicode"
)

type tokenizerJSON struct {
	Version string `json:"version"`
	Model   struct {
		Type    string              `json:"type"`
		Vocab   map[string]int      `json:"vocab"`
		Merges  json.RawMessage     `json:"merges"`
		BosTok  *string             `json:"bos_token"`
		EosTok  *string             `json:"eos_token"`
	} `json:"model"`
	AddedTokens []struct {
		ID    int    `json:"id"`
		Token string `json:"content"`
	} `json:"added_tokens"`
}

// Tokenizer does byte-level BPE over Qwen's pre-tokenization.
type Tokenizer struct {
	vocab  map[string]int   // token string -> id
	byID   map[int]string   // id -> token string
	merges map[string]int   // "a\x00b" -> rank
	special map[string]int  // added special tokens
}

// Load parses a tokenizer.json file.
func Load(path string) (*Tokenizer, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	var tj tokenizerJSON
	if err := json.NewDecoder(bufio.NewReader(f)).Decode(&tj); err != nil {
		return nil, fmt.Errorf("tokenizer: %w", err)
	}
	t := &Tokenizer{
		vocab:   tj.Model.Vocab,
		byID:    make(map[int]string, len(tj.Model.Vocab)),
		merges:  make(map[string]int),
		special: make(map[string]int),
	}
	if t.vocab == nil {
		return nil, fmt.Errorf("tokenizer.json: no vocab")
	}
	for tok, id := range t.vocab {
		t.byID[id] = tok
	}
	// merges: list of ["a","b"] (>=1.1 format) or "a b" strings
	var pairs [][2]string
	if len(tj.Model.Merges) > 0 && tj.Model.Merges[0] == '[' {
		if err := json.Unmarshal(tj.Model.Merges, &pairs); err != nil {
			return nil, fmt.Errorf("tokenizer: merges: %w", err)
		}
	} else {
		var strs []string
		if err := json.Unmarshal(tj.Model.Merges, &strs); err != nil {
			return nil, fmt.Errorf("tokenizer: merges: %w", err)
		}
		for _, s := range strs {
			if i := strings.Index(s, " "); i >= 0 {
				pairs = append(pairs, [2]string{s[:i], s[i+1:]})
			}
		}
	}
	for i, p := range pairs {
		t.merges[p[0]+"\x00"+p[1]] = i
	}
	for _, at := range tj.AddedTokens {
		t.special[at.Token] = at.ID
		if _, ok := t.vocab[at.Token]; !ok {
			t.vocab[at.Token] = at.ID
			t.byID[at.ID] = at.Token
		}
	}
	return t, nil
}

// ID resolves a token string (special or vocab) to its id.
func (t *Tokenizer) ID(tok string) (int, bool) {
	id, ok := t.vocab[tok]
	return id, ok
}

// Decode turns ids back into text. Byte-level tokens are remapped back to
// raw bytes; special tokens pass through literally.
func (t *Tokenizer) Decode(ids []int) string {
	var b strings.Builder
	for _, id := range ids {
		s, ok := t.byID[id]
		if !ok {
			continue
		}
		if _, isSpecial := t.special[s]; isSpecial {
			b.WriteString(s)
			continue
		}
		b.WriteString(unicodeToBytes(s))
	}
	return b.String()
}

// Encode runs pre-tokenization + BPE on raw text.
func (t *Tokenizer) Encode(text string) []int {
	var ids []int
	for _, piece := range preTokenize(text) {
		ids = append(ids, t.bpe(piece)...)
	}
	return ids
}

// bpe encodes one pre-token (already matched) into ids.
func (t *Tokenizer) bpe(piece string) []int {
	// map raw bytes to the unicode-remapped symbols the vocab uses
	syms := make([]string, 0, len(piece))
	for i := 0; i < len(piece); i++ {
		syms = append(syms, byteToUnicode(piece[i]))
	}
	for len(syms) > 1 {
		bestRank, bestIdx := -1, -1
		for i := 0; i+1 < len(syms); i++ {
			if r, ok := t.merges[syms[i]+"\x00"+syms[i+1]]; ok && (bestRank < 0 || r < bestRank) {
				bestRank, bestIdx = r, i
			}
		}
		if bestIdx < 0 {
			break
		}
		syms = append(syms[:bestIdx], syms[bestIdx]+syms[bestIdx+1])
	}
	out := make([]int, 0, len(syms))
	for _, s := range syms {
		if id, ok := t.vocab[s]; ok {
			out = append(out, id)
		}
		// unknown symbol: skip (can't happen with a complete byte vocab)
	}
	return out
}

// EncodeWithSpecial encodes text, honoring known special tokens that appear
// literally in it (e.g. <|im_start|>).
func (t *Tokenizer) EncodeWithSpecial(text string) []int {
	// find the longest special token at each position, encode the gaps
	toks := make([]string, 0, len(t.special))
	for tok := range t.special {
		toks = append(toks, tok)
	}
	sort.Slice(toks, func(i, j int) bool { return len(toks[i]) > len(toks[j]) })

	var ids []int
	for i := 0; i < len(text); {
		matched := ""
		for _, tok := range toks {
			if strings.HasPrefix(text[i:], tok) {
				matched = tok
				break
			}
		}
		if matched != "" {
			ids = append(ids, t.special[matched])
			i += len(matched)
			continue
		}
		j := i
		for j < len(text) {
			m2 := ""
			for _, tok := range toks {
				if strings.HasPrefix(text[j:], tok) {
					m2 = tok
					break
				}
			}
			if m2 != "" {
				break
			}
			j++
		}
		ids = append(ids, t.Encode(text[i:j])...)
		i = j
	}
	return ids
}

// ---------------------------------------------------------------------------
// Qwen2-style pre-tokenizer, hand-rolled (Go regexp has no lookaheads).
//
// The HF pattern, in order:
//   contractions ('s 't 're 've 'm 'll 'd 'll 've — case-insensitive)
//   [^\r\n\p{L}\p{N}]?\p{L}+          (optional single punct/space + letters)
//   \p{N}{1,3}                        (numbers, max 3 per token)
//    ?[^\s\p{L}\p{N}]+[\r\n]*         (punct runs, optional lead space)
//   \s*[\r\n]+                        (newlines with lead blanks)
//   \s+(?!\S)                         (blanks, all but the last)
//   \s+                               (leftover blanks)
func preTokenize(text string) []string {
	var out []string
	rs := []rune(text)
	n := len(rs)
	i := 0
	isLetter := func(r rune) bool { return unicode.IsLetter(r) }
	isNumber := func(r rune) bool { return unicode.IsNumber(r) }
	isSpace := func(r rune) bool { return unicode.IsSpace(r) }

	for i < n {
		r := rs[i]
		switch {
		case isSpace(r):
			// \s*[\r\n]+ first: blanks followed by newline(s) fold together
			j := i
			for j < n && isSpace(rs[j]) && rs[j] != '\r' && rs[j] != '\n' {
				j++
			}
			if j < n && (rs[j] == '\r' || rs[j] == '\n') {
				for j < n && (rs[j] == '\r' || rs[j] == '\n') {
					j++
				}
				out = append(out, string(rs[i:j]))
				i = j
				continue
			}
			// \s+(?!\S): consume blanks, all but the last if a word follows
			j = i
			for j < n && isSpace(rs[j]) {
				j++
			}
			end := j
			if j < n && j-i > 0 {
				end = j - 1 // last blank goes to the next word
			}
			if end > i {
				out = append(out, string(rs[i:end]))
			}
			// leftover single blank: attach to next letters / punct run /
			// number per the ` ?X` alternatives — handled by the next loop
			// iteration seeing a space we deliberately re-examine.
			i = end
		default:
			// contractions (ASCII, case-insensitive)
			if r == '\'' {
				if c, l := matchContraction(rs[i:]); l > 0 {
					out = append(out, string(rs[i:i+l]))
					i += l
					continue
				}
			}
			// [^\r\n\p{L}\p{N}]?\p{L}+ — optional one non-letter prefix
			j := i
			if !isLetter(r) && r != '\r' && r != '\n' && !isNumber(r) &&
				i+1 < n && isLetter(rs[i+1]) {
				j++
			}
			if j < n && isLetter(rs[j]) {
				for j < n && isLetter(rs[j]) {
					j++
				}
				out = append(out, string(rs[i:j]))
				i = j
				continue
			}
			if isNumber(r) {
				j := i
				for j < n && j-i < 3 && isNumber(rs[j]) {
					j++
				}
				out = append(out, string(rs[i:j]))
				i = j
				continue
			}
			if !isSpace(r) {
				//  ?[^\s\p{L}\p{N}]+[\r\n]*
				j = i
				for j < n && !isSpace(rs[j]) && !isLetter(rs[j]) && !isNumber(rs[j]) {
					j++
				}
				for j < n && (rs[j] == '\r' || rs[j] == '\n') {
					j++
				}
				out = append(out, string(rs[i:j]))
				i = j
				continue
			}
			// lone space with nothing usable after: emit it
			out = append(out, string(r))
			i++
		}
	}
	return out
}

func matchContraction(rs []rune) (int, int) {
	lower := func(r rune) rune { return unicode.ToLower(r) }
	if len(rs) < 2 {
		return 0, 0
	}
	switch lower(rs[1]) {
	case 's', 't', 'm', 'd':
		return 0, 2
	case 'r':
		if len(rs) >= 3 && lower(rs[2]) == 'e' {
			return 0, 3
		}
	case 'v':
		if len(rs) >= 3 && lower(rs[2]) == 'e' {
			return 0, 3
		}
	case 'l':
		if len(rs) >= 3 && lower(rs[2]) == 'l' {
			return 0, 3
		}
	}
	return 0, 0
}

// byteToUnicode maps a raw byte to the printable unicode symbol GPT-2/Qwen
// vocabs use (the classic bytes_to_unicode table, computed on the fly).
func byteToUnicode(b byte) string {
	return string(bytesToUnicodeRune(b))
}

func bytesToUnicodeRune(b byte) rune {
	switch {
	case b >= '!' && b <= '~', b >= 0xA1 && b <= 0xAC, b >= 0xAE && b <= 0xFF:
		return rune(b)
	}
	// everything else is mapped into 0x100+n in order
	n := 0
	for c := 0; c < 256; c++ {
		if (c >= '!' && c <= '~') || (c >= 0xA1 && c <= 0xAC) || (c >= 0xAE && c <= 0xFF) {
			continue
		}
		if c == int(b) {
			return rune(0x100 + n)
		}
		n++
	}
	return rune(0x100 + n)
}

// unicodeToBytes inverts bytesToUnicodeRune for decoding.
func unicodeToBytes(s string) string {
	var out []byte
	for _, r := range s {
		out = append(out, unicodeToByte(r)...)
	}
	return string(out)
}

func unicodeToByte(r rune) []byte {
	if r < 256 && ((r >= '!' && r <= '~') || (r >= 0xA1 && r <= 0xAC) || (r >= 0xAE && r <= 0xFF)) {
		return []byte{byte(r)}
	}
	n := 0
	for c := 0; c < 256; c++ {
		if (c >= '!' && c <= '~') || (c >= 0xA1 && c <= 0xAC) || (c >= 0xAE && c <= 0xFF) {
			continue
		}
		if rune(0x100+n) == r {
			return []byte{byte(c)}
		}
		n++
	}
	return []byte(string(rune(r))) // unreachable in practice
}
