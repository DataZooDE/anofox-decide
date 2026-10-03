// DecideTokenizer — HF tokenizers BPE port (Julia-1). See decide_tokenizer.hpp.

#include "decide_tokenizer.hpp"
#include "decide_errors.hpp"

#include "yyjson.hpp"
#include "utf8proc.hpp"
#include "utf8proc_wrapper.hpp"

#include "duckdb/common/exception.hpp"

#include <algorithm>
#include <cstring>
#include <fstream>
#include <functional>
#include <limits>
#include <sstream>

namespace duckdb {
namespace anofox {

using namespace duckdb_yyjson; // NOLINT (tabfm_manifest.cpp precedent)

namespace {

//--- UTF-8 -------------------------------------------------------------------

// Decode one codepoint; returns bytes consumed (0 on invalid).
size_t Utf8DecodeOne(const char *s, size_t avail, char32_t &cp) {
	unsigned char c = s[0];
	if (c < 0x80) {
		cp = c;
		return 1;
	}
	size_t need = 0;
	char32_t v = 0;
	if ((c & 0xE0) == 0xC0) {
		need = 2;
		v = c & 0x1F;
	} else if ((c & 0xF0) == 0xE0) {
		need = 3;
		v = c & 0x0F;
	} else if ((c & 0xF8) == 0xF0) {
		need = 4;
		v = c & 0x07;
	} else {
		return 0;
	}
	if (avail < need) {
		return 0;
	}
	for (size_t i = 1; i < need; i++) {
		unsigned char d = s[i];
		if ((d & 0xC0) != 0x80) {
			return 0;
		}
		v = (v << 6) | (d & 0x3F);
	}
	cp = v;
	return need;
}

std::string Utf8EncodeOne(char32_t cp) {
	std::string out;
	if (cp < 0x80) {
		out.push_back((char)cp);
	} else if (cp < 0x800) {
		out.push_back((char)(0xC0 | (cp >> 6)));
		out.push_back((char)(0x80 | (cp & 0x3F)));
	} else if (cp < 0x10000) {
		out.push_back((char)(0xE0 | (cp >> 12)));
		out.push_back((char)(0x80 | ((cp >> 6) & 0x3F)));
		out.push_back((char)(0x80 | (cp & 0x3F)));
	} else {
		out.push_back((char)(0xF0 | (cp >> 18)));
		out.push_back((char)(0x80 | ((cp >> 12) & 0x3F)));
		out.push_back((char)(0x80 | ((cp >> 6) & 0x3F)));
		out.push_back((char)(0x80 | (cp & 0x3F)));
	}
	return out;
}

// SentencePiece-style byte fallback (Gemma/mmBERT vocab): byte b is the
// literal token "<0xXX>", uppercase hex.
void InitByteEncoder(std::string out[256]) {
	static const char *hex = "0123456789ABCDEF";
	for (int b = 0; b < 256; b++) {
		out[b] = std::string("<0x") + hex[b >> 4] + hex[b & 15] + ">";
	}
}

// GPT-2 bytes_to_unicode alphabet (ByteLevel BPE): printable ASCII +
// Latin-1 supplement map to themselves; every other byte maps to U+0100 up.
void InitGpt2ByteEncoder(std::string out[256]) {
	std::vector<int> bs;
	for (int b = 33; b < 127; b++) {
		bs.push_back(b);
	}
	for (int b = 161; b < 173; b++) {
		bs.push_back(b);
	}
	for (int b = 174; b < 256; b++) {
		bs.push_back(b);
	}
	std::vector<int> cs = bs;
	int n = 0;
	for (int b = 0; b < 256; b++) {
		bool in_bs = false;
		for (int x : bs) {
			if (x == b) {
				in_bs = true;
				break;
			}
		}
		if (!in_bs) {
			bs.push_back(b);
			cs.push_back(256 + n);
			n++;
		}
	}
	for (size_t i = 0; i < bs.size(); i++) {
		out[bs[i]] = Utf8EncodeOne((char32_t)cs[i]);
	}
}

// Rust regex \s (Unicode White_Space).
bool IsSpaceCp(char32_t c) {
	return (c >= 0x09 && c <= 0x0D) || c == 0x20 || c == 0x85 || c == 0xA0 || c == 0x1680 ||
	       (c >= 0x2000 && c <= 0x200A) || c == 0x2028 || c == 0x2029 || c == 0x202F || c == 0x205F || c == 0x3000;
}
bool IsLetterCp(char32_t c) {
	auto cat = utf8proc_category((utf8proc_int32_t)c);
	return cat >= UTF8PROC_CATEGORY_LU && cat <= UTF8PROC_CATEGORY_LO;
}
bool IsNumberCp(char32_t c) {
	auto cat = utf8proc_category((utf8proc_int32_t)c);
	return cat >= UTF8PROC_CATEGORY_ND && cat <= UTF8PROC_CATEGORY_NO;
}

// GPT-2 pre-tokenizer (HF ByteLevel use_regex):
//   's|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+
std::vector<std::string> Gpt2Split(const std::string &text) {
	std::vector<char32_t> cps;
	std::vector<size_t> off; // byte offset of each codepoint, plus end
	for (size_t i = 0; i < text.size();) {
		char32_t cp = 0;
		size_t n = Utf8DecodeOne(text.c_str() + i, text.size() - i, cp);
		if (!n) {
			cp = (unsigned char)text[i];
			n = 1;
		}
		cps.push_back(cp);
		off.push_back(i);
		i += n;
	}
	off.push_back(text.size());
	const size_t n = cps.size();
	std::vector<std::string> out;
	auto emit = [&](size_t from, size_t to) { out.push_back(text.substr(off[from], off[to] - off[from])); };
	size_t i = 0;
	while (i < n) {
		if (cps[i] == '\'') {
			static const char *suffixes[] = {"s", "t", "re", "ve", "m", "ll", "d"};
			bool hit = false;
			for (auto suf : suffixes) {
				size_t len = strlen(suf);
				if (i + len < n + 0 && i + 1 + len <= n) {
					bool ok = true;
					for (size_t k = 0; k < len; k++) {
						if (cps[i + 1 + k] != (char32_t)suf[k]) {
							ok = false;
							break;
						}
					}
					if (ok) {
						emit(i, i + 1 + len);
						i += 1 + len;
						hit = true;
						break;
					}
				}
			}
			if (hit) {
				continue;
			}
		}
		size_t k = (cps[i] == ' ') ? i + 1 : i;
		if (k < n) {
			size_t j = k;
			if (IsLetterCp(cps[k])) {
				while (j < n && IsLetterCp(cps[j])) {
					j++;
				}
			} else if (IsNumberCp(cps[k])) {
				while (j < n && IsNumberCp(cps[j])) {
					j++;
				}
			} else if (!IsSpaceCp(cps[k])) {
				while (j < n && !IsSpaceCp(cps[j]) && !IsLetterCp(cps[j]) && !IsNumberCp(cps[j])) {
					j++;
				}
			}
			if (j > k) {
				emit(i, j);
				i = j;
				continue;
			}
		}
		// Whitespace run: \s+(?!\S) leaves the last space for the next word.
		size_t r = i;
		while (r < n && IsSpaceCp(cps[r])) {
			r++;
		}
		size_t take = (r == n || r - i == 1) ? r : r - 1;
		emit(i, take);
		i = take;
	}
	return out;
}

std::string ToUtf8Str(yyjson_val *v) {
	return std::string(yyjson_get_str(v), yyjson_get_len(v));
}

} // namespace

void DecideTokenizer::Load(const std::string &tokenizer_json_path, const std::string &cls_content,
                           const std::string &sep_content, const std::string &mask_content,
                           const std::string &pad_content, const std::string &unk_content) {
	// Plain stdio on purpose (not DuckDB's FileSystem): tokenizer loading is
	// context-free so the Catch2 golden tests run without a database.
	std::ifstream in(tokenizer_json_path, std::ios::binary);
	if (!in) {
		throw InvalidInputException(DecideMsg(
		    function_name, "tokenizer file not found: '" + tokenizer_json_path + "'",
		    "pass the tokenizer.json that shipped with the model"));
	}
	std::ostringstream ss;
	ss << in.rdbuf();
	Parse(ss.str(), tokenizer_json_path, cls_content, sep_content, mask_content, pad_content, unk_content);
}

void DecideTokenizer::Parse(const std::string &tokenizer_json, const std::string &tokenizer_json_path,
                            const std::string &cls_content, const std::string &sep_content,
                            const std::string &mask_content, const std::string &pad_content,
                            const std::string &unk_content) {
	const std::string &raw = tokenizer_json;
	// "the tokenizer file 'x' <what>. Fix: ..." in the project's error shape.
	auto bad = [&](const std::string &what) {
		return InvalidInputException(DecideMsg(function_name, "the tokenizer file '" + tokenizer_json_path + "' " + what,
		                                       "pass the tokenizer.json of a Julia-1 or Laya checkpoint (the file "
		                                       "named tokenizer.json, not tokenizer_config.json or a vocabulary file)"));
	};
	struct DocFree {
		yyjson_doc *d;
		~DocFree() {
			if (d) {
				yyjson_doc_free(d);
			}
		}
	} doc {yyjson_read(raw.c_str(), raw.size(), 0)};
	if (!doc.d) {
		throw bad("is not valid JSON");
	}
	auto root = yyjson_doc_get_root(doc.d);
	if (!root || !yyjson_is_obj(root)) {
		throw bad("is not a JSON object");
	}
	auto model = yyjson_obj_get(root, "model");
	if (!model || !yyjson_is_obj(model)) {
		throw bad("has no 'model' object");
	}
	auto type_v = yyjson_obj_get(model, "type");
	if (!type_v || !yyjson_is_str(type_v) || ToUtf8Str(type_v) != "BPE") {
		const std::string kind = (type_v && yyjson_is_str(type_v)) ? ToUtf8Str(type_v) : std::string("unknown");
		throw InvalidInputException(DecideMsg(
		    function_name,
		    "the tokenizer file '" + tokenizer_json_path + "' is a " + kind + " tokenizer, but local models need a BPE "
		    "tokenizer (the one Julia-1 and Laya checkpoints ship)",
		    "pass the tokenizer.json of a Julia-1 or Laya checkpoint"));
	}
	vocab.clear();
	merge_rank.clear();
	specials_by_content.clear();

	auto vocab_v = yyjson_obj_get(model, "vocab");
	if (!vocab_v || !yyjson_is_obj(vocab_v)) {
		throw bad("has no model.vocab object");
	}
	yyjson_obj_iter viter;
	yyjson_obj_iter_init(vocab_v, &viter);
	yyjson_val *vkey;
	while ((vkey = yyjson_obj_iter_next(&viter))) {
		auto vval = yyjson_obj_iter_get_val(vkey);
		if (!yyjson_is_uint(vval) && !yyjson_is_sint(vval)) {
			throw bad("has a vocabulary entry whose id is not an integer");
		}
		std::string token(yyjson_get_str(vkey), yyjson_get_len(vkey));
		vocab[token] = (int32_t)yyjson_get_sint(vval);
	}
	vocab_size = vocab.size();

	auto merges_v = yyjson_obj_get(model, "merges");
	if (merges_v && yyjson_is_arr(merges_v)) {
		yyjson_val *entry;
		yyjson_arr_iter miter;
		yyjson_arr_iter_init(merges_v, &miter);
		int rank = 0;
		while ((entry = yyjson_arr_iter_next(&miter))) {
			// Entries are either "left right" strings or [left, right] pairs.
			std::string left, right;
			if (yyjson_is_str(entry)) {
				std::string s = ToUtf8Str(entry);
				auto sp = s.find(' ');
				if (sp == std::string::npos) {
					throw bad("has a malformed merge entry");
				}
				left = s.substr(0, sp);
				right = s.substr(sp + 1);
			} else if (yyjson_is_arr(entry) && yyjson_arr_size(entry) == 2) {
				yyjson_val *it;
				yyjson_arr_iter piter;
				yyjson_arr_iter_init(entry, &piter);
				it = yyjson_arr_iter_next(&piter);
				if (!it || !yyjson_is_str(it)) {
					throw bad("has a malformed merge pair");
				}
				left = ToUtf8Str(it);
				it = yyjson_arr_iter_next(&piter);
				if (!it || !yyjson_is_str(it)) {
					throw bad("has a malformed merge pair");
				}
				right = ToUtf8Str(it);
			} else {
				throw bad("has a malformed merge entry");
			}
			merge_rank[left + "\x1f" + right] = rank++;
		}
	}

	auto bf_v = yyjson_obj_get(model, "byte_fallback");
	byte_fallback = !bf_v || !yyjson_is_bool(bf_v) || yyjson_get_bool(bf_v);
	// ByteLevel pre-tokenizer => GPT-2 family (no byte fallback tokens).
	byte_level = false;
	auto pre_v = yyjson_obj_get(root, "pre_tokenizer");
	if (pre_v && yyjson_is_obj(pre_v)) {
		auto pt = yyjson_obj_get(pre_v, "type");
		byte_level = pt && yyjson_is_str(pt) && ToUtf8Str(pt) == "ByteLevel";
	}
	if (byte_level) {
		byte_fallback = false;
		InitGpt2ByteEncoder(byte_to_token);
	} else {
		InitByteEncoder(byte_to_token);
	}
	added_raw.clear();
	added_norm.clear();

	auto added_v = yyjson_obj_get(root, "added_tokens");
	if (added_v && yyjson_is_arr(added_v)) {
		yyjson_val *entry;
		yyjson_arr_iter aiter;
		yyjson_arr_iter_init(added_v, &aiter);
		while ((entry = yyjson_arr_iter_next(&aiter))) {
			if (!entry || !yyjson_is_obj(entry)) {
				continue;
			}
			auto id_v = yyjson_obj_get(entry, "id");
			auto content_v = yyjson_obj_get(entry, "content");
			auto special_v = yyjson_obj_get(entry, "special");
			auto normalized_v = yyjson_obj_get(entry, "normalized");
			if (!id_v || !content_v || !yyjson_is_str(content_v)) {
				continue;
			}
			bool special = special_v && yyjson_is_bool(special_v) && yyjson_get_bool(special_v);
			bool normalized = normalized_v && yyjson_is_bool(normalized_v) && yyjson_get_bool(normalized_v);
			// Only non-normalized specials split the text (HF behavior for
			// special tokens); normalized ones flow through BPE.
			if (special && !normalized) {
				specials_by_content.emplace_back(ToUtf8Str(content_v),
				                                 (int32_t)yyjson_get_sint(id_v));
			}
			if (byte_level) {
				// HF extracts every added token (special or not) before
				// pre-tokenization; normalized ones match post-NFC text.
				(normalized ? added_norm : added_raw)
				    .emplace_back(ToUtf8Str(content_v), (int32_t)yyjson_get_sint(id_v));
			}
		}
		std::sort(specials_by_content.begin(), specials_by_content.end(),
		          [](const auto &a, const auto &b) { return a.first.size() > b.first.size(); });
		auto by_len = [](const auto &a, const auto &b) { return a.first.size() > b.first.size(); };
		std::sort(added_raw.begin(), added_raw.end(), by_len);
		std::sort(added_norm.begin(), added_norm.end(), by_len);
	}

	auto resolve = [&](const std::string &content, const char *role) {
		for (auto &kv : specials_by_content) {
			if (kv.first == content) {
				return kv.second;
			}
		}
		auto it = vocab.find(content);
		if (it != vocab.end()) {
			return it->second;
		}
		throw bad(std::string("has no '") + content + "' token, which a local model needs as its " + role +
		          " token (this tokenizer belongs to another model family)");
	};
	// ByteLevel tokenizers name their specials [CLS]/[SEP]/...: swap in
	// those names when the caller kept the Gemma-style defaults.
	const bool gemma_defaults = cls_content == "<bos>" && sep_content == "<eos>" && mask_content == "<mask>" &&
	                            pad_content == "<pad>" && unk_content == "<unk>";
	const std::string cls_n = (byte_level && gemma_defaults) ? "[CLS]" : cls_content;
	const std::string sep_n = (byte_level && gemma_defaults) ? "[SEP]" : sep_content;
	const std::string mask_n = (byte_level && gemma_defaults) ? "[MASK]" : mask_content;
	const std::string pad_n = (byte_level && gemma_defaults) ? "[PAD]" : pad_content;
	const std::string unk_n = (byte_level && gemma_defaults) ? "[UNK]" : unk_content;
	unk_token = unk_n;
	specials.cls = resolve(cls_n, "cls");
	specials.sep = resolve(sep_n, "sep");
	specials.mask = resolve(mask_n, "mask");
	specials.pad = resolve(pad_n, "pad");
	specials.unk = resolve(unk_n, "unk");
	specials.mask_token = mask_n;
}

std::vector<int32_t> DecideTokenizer::BpeEncodeWord(const std::vector<std::string> &chars) const {
	// Split unknown characters into byte tokens first when byte_fallbackimis
	// enabled (HF BPE byte fallback), then greedy lowest-rank merges.
	std::vector<std::string> word;
	for (auto &ch : chars) {
		if (vocab.find(ch) != vocab.end() || !byte_fallback) {
			word.push_back(ch);
			continue;
		}
		for (unsigned char b : ch) {
			word.push_back(byte_to_token[b]);
		}
	}
	while (word.size() > 1) {
		int best_rank = std::numeric_limits<int>::max();
		size_t best_at = 0;
		bool found = false;
		for (size_t i = 0; i + 1 < word.size(); i++) {
			auto it = merge_rank.find(word[i] + "\x1f" + word[i + 1]);
			if (it != merge_rank.end() && it->second < best_rank) {
				best_rank = it->second;
				best_at = i;
				found = true;
			}
		}
		if (!found) {
			break;
		}
		// Merge every non-overlapping occurrence of the best pair, left to
		// right (HF tokenizers BPE semantics).
		std::string left = word[best_at];
		std::string right = word[best_at + 1];
		std::vector<std::string> merged;
		for (size_t i = 0; i < word.size();) {
			if (i + 1 < word.size() && word[i] == left && word[i + 1] == right) {
				merged.push_back(left + right);
				i += 2;
			} else {
				merged.push_back(word[i]);
				i += 1;
			}
		}
		word.swap(merged);
	}
	std::vector<int32_t> ids;
	for (auto &tok : word) {
		auto it = vocab.find(tok);
		ids.push_back(it != vocab.end() ? it->second : specials.unk);
	}
	return ids;
}

std::vector<int32_t> DecideTokenizer::EncodeByteLevel(const std::string &text) const {
	std::vector<int32_t> ids;
	// Leftmost-longest split of `s` on `tokens`; plain spans go to `plain`.
	auto split = [](const std::string &s, const std::vector<std::pair<std::string, int32_t>> &tokens,
	                const std::function<void(const std::string &)> &plain,
	                const std::function<void(int32_t)> &token) {
		std::string span;
		size_t pos = 0;
		while (pos < s.size()) {
			size_t best_len = 0;
			int32_t best_id = 0;
			for (auto &kv : tokens) {
				if (kv.first.size() > best_len && s.compare(pos, kv.first.size(), kv.first) == 0) {
					best_len = kv.first.size();
					best_id = kv.second;
				}
			}
			if (best_len > 0) {
				if (!span.empty()) {
					plain(span);
					span.clear();
				}
				token(best_id);
				pos += best_len;
			} else {
				span.push_back(s[pos++]);
			}
		}
		if (!span.empty()) {
			plain(span);
		}
	};
	auto encode_plain = [&](const std::string &seg) {
		for (auto &piece : Gpt2Split(seg)) {
			std::vector<std::string> chars;
			for (unsigned char b : piece) {
				chars.push_back(byte_to_token[b]);
			}
			auto piece_ids = BpeEncodeWord(chars);
			ids.insert(ids.end(), piece_ids.begin(), piece_ids.end());
		}
	};
	auto normalize_and_split = [&](const std::string &seg) {
		std::string norm = seg;
		if (Utf8Proc::IsValid(seg.c_str(), seg.size())) {
			auto *n = Utf8Proc::Normalize(seg.c_str(), seg.size()); // NFC
			if (n) {
				norm = n;
				free(n);
			}
		}
		split(norm, added_norm, encode_plain, [&](int32_t id) { ids.push_back(id); });
	};
	split(text, added_raw, normalize_and_split, [&](int32_t id) { ids.push_back(id); });
	return ids;
}

std::vector<int32_t> DecideTokenizer::Encode(const std::string &text) const {
	if (byte_level) {
		return EncodeByteLevel(text);
	}
	static const std::string kWordBoundary = "\xe2\x96\x81"; // U+2581
	std::vector<int32_t> ids;
	// 1. Longest-match split on special tokens (never normalized/BPE'd).
	// 2. Normalizer Replace(" " -> U+2581) on the remaining spans.
	// 3. Metaspace split (prepend always): pieces keep their leading U+2581.
	// 4. BPE per piece over Unicode characters.
	size_t pos = 0;
	std::string span;
	auto flush_span = [&]() {
		if (span.empty()) {
			return;
		}
		std::string norm;
		for (size_t i = 0; i < span.size();) {
			if (span[i] == ' ') {
				norm += kWordBoundary;
				i++;
			} else {
				norm.push_back(span[i]);
				i++;
			}
		}
		// Metaspace prepend_scheme=always: the text starts with a boundary so
		// the first piece keeps its leading U+2581 (HF emits "▁A", not "A").
		if (norm.compare(0, kWordBoundary.size(), kWordBoundary) != 0) {
			norm = kWordBoundary + norm;
		}
		// Split into metaspace pieces.
		std::vector<std::string> pieces;
		std::string cur;
		bool first = true;
		for (size_t i = 0; i < norm.size();) {
			bool boundary = norm.compare(i, kWordBoundary.size(), kWordBoundary) == 0;
			if (boundary && !first) {
				pieces.push_back(cur);
				cur.clear();
			}
			size_t step = boundary ? kWordBoundary.size() : 1;
			// Advance by whole UTF-8 characters for non-boundary bytes.
			if (!boundary) {
				char32_t cp = 0;
				size_t n = Utf8DecodeOne(norm.c_str() + i, norm.size() - i, cp);
				step = n ? n : 1;
			}
			cur.append(norm, i, step);
			i += step;
			first = false;
		}
		if (!cur.empty()) {
			pieces.push_back(cur);
		}
		for (auto &piece : pieces) {
			// Piece -> Unicode characters.
			std::vector<std::string> chars;
			for (size_t i = 0; i < piece.size();) {
				char32_t cp = 0;
				size_t n = Utf8DecodeOne(piece.c_str() + i, piece.size() - i, cp);
				if (!n) {
					n = 1;
				}
				chars.push_back(piece.substr(i, n));
				i += n;
			}
			auto piece_ids = BpeEncodeWord(chars);
			ids.insert(ids.end(), piece_ids.begin(), piece_ids.end());
		}
		span.clear();
	};
	while (pos < text.size()) {
		size_t best_len = 0;
		int32_t best_id = 0;
		for (auto &kv : specials_by_content) {
			if (kv.first.size() > best_len && text.compare(pos, kv.first.size(), kv.first) == 0) {
				best_len = kv.first.size();
				best_id = kv.second;
			}
		}
		if (best_len > 0) {
			flush_span();
			ids.push_back(best_id);
			pos += best_len;
		} else {
			span.push_back(text[pos]);
			pos++;
		}
	}
	flush_span();
	return ids;
}

} // namespace anofox
} // namespace duckdb
