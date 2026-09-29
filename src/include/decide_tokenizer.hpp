#pragma once

// DecideTokenizer — HF tokenizers BPE port for the Julia-1 tokenizer.
//
// Replicates, for encode(add_special_tokens=false):
//   normalizer Replace(" " -> U+2581), Metaspace pretokenizer (prepend
//   always, split), BPE merge ranks, byte_fallback, added/special token
//   longest-match splitting. fuse_unk and offsets are decode-side concerns
//   and do not affect emitted ids.
//
// Special ids (cls/sep/mask/pad) resolve by content through added_tokens
// using tokenizer_config.json names. NOTE: the live HF tokenizer reports
// cls=2 (<bos>), NOT the encoder config's cls_token_id=1 — the golden file
// (test/fixtures/julia_tokenizer_golden.json) is authoritative; see
// docs/julia-1-spec.json.

#include <cstdint>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace duckdb {
namespace anofox {

struct DecideSpecialIds {
	int32_t cls = -1;
	int32_t sep = -1;
	int32_t mask = -1;
	int32_t pad = -1;
	int32_t unk = -1;
	std::string mask_token;
};

class DecideTokenizer {
public:
	DecideTokenizer() = default;

	//! Load tokenizer.json + resolve specials by content strings.
	//! Throws InvalidInputException naming the file on any schema problem.
	//! Plain stdio on purpose (not DuckDB's FileSystem): tokenizer loading
	//! is context-free so the Catch2 golden tests run without a database.
	//! Production reads go through DecideReadLocalFile + Parse below.
	void Load(const std::string &tokenizer_json_path, const std::string &cls_content = "<bos>",
	          const std::string &sep_content = "<eos>", const std::string &mask_content = "<mask>",
	          const std::string &pad_content = "<pad>", const std::string &unk_content = "<unk>");
	//! Parse already-read bytes (same schema handling as Load; pure).
	void Parse(const std::string &tokenizer_json, const std::string &tokenizer_json_path,
	           const std::string &cls_content = "<bos>", const std::string &sep_content = "<eos>",
	           const std::string &mask_content = "<mask>", const std::string &pad_content = "<pad>",
	           const std::string &unk_content = "<unk>");

	bool Loaded() const {
		return !vocab.empty();
	}

	const DecideSpecialIds &Specials() const {
		return specials;
	}

	size_t VocabSize() const {
		return vocab_size;
	}

	//! Encode without special tokens (data.sequence only needs this form).
	std::vector<int32_t> Encode(const std::string &text) const;

private:
	std::string piece_to_id_token(const std::string &piece, bool &known) const;
	std::vector<int32_t> BpeEncodeWord(const std::vector<std::string> &chars) const;

	// token -> id (UTF-8 encoded token strings as in tokenizer.json)
	std::unordered_map<std::string, int32_t> vocab;
	// "left\x1fright" -> merge rank (lower merges first)
	std::unordered_map<std::string, int> merge_rank;
	// byte (0-255) -> single-char unicode token (HF bytes encoder)
	std::string byte_to_token[256];
	// special content -> id (longest match wins at split time)
	std::vector<std::pair<std::string, int32_t>> specials_by_content;
	bool byte_fallback = true;
	DecideSpecialIds specials;
	std::string unk_token = "<unk>";
	size_t vocab_size = 0;
};

} // namespace anofox
} // namespace duckdb
