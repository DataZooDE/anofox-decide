#include "decide_mlx_weights.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <sys/stat.h>

namespace anofox {
namespace mlxgraph {
namespace {

[[noreturn]] void Fail(const std::string &what) {
	throw std::runtime_error("anofox_decide mlx weights: " + what);
}

//! A JSON value, just enough for the tensor map and the safetensors header: numbers are doubles (offsets stay
//! far below 2^53), objects keep their members in a map.
struct Json {
	enum Type { Null, Bool, Number, String, Array, Object } type = Null;
	double number = 0;
	std::string str;
	std::vector<Json> array;
	std::map<std::string, Json> object;

	const Json *Find(const std::string &key) const {
		if (type != Object) {
			return nullptr;
		}
		auto it = object.find(key);
		return it == object.end() ? nullptr : &it->second;
	}
};

class Parser {
public:
	Parser(const char *data, size_t size, std::string source)
	    : p_(data), end_(data + size), source_(std::move(source)) {
	}

	Json Parse() {
		Json v = Value(0);
		Skip();
		if (p_ != end_) {
			Bad("trailing characters after the value");
		}
		return v;
	}

private:
	const char *p_;
	const char *end_;
	std::string source_;

	[[noreturn]] void Bad(const char *what) const {
		Fail(source_ + ": malformed JSON (" + what + ")");
	}
	void Skip() {
		while (p_ < end_ && (*p_ == ' ' || *p_ == '\n' || *p_ == '\t' || *p_ == '\r')) {
			p_++;
		}
	}
	bool Eat(char c) {
		Skip();
		if (p_ < end_ && *p_ == c) {
			p_++;
			return true;
		}
		return false;
	}
	bool Literal(const char *word) {
		const size_t n = std::strlen(word);
		if (static_cast<size_t>(end_ - p_) >= n && std::strncmp(p_, word, n) == 0) {
			p_ += n;
			return true;
		}
		return false;
	}
	std::string String() {
		std::string out;
		if (p_ >= end_ || *p_ != '"') {
			Bad("expected a string");
		}
		p_++;
		while (p_ < end_ && *p_ != '"') {
			if (*p_ != '\\') {
				out += *p_++;
				continue;
			}
			p_++;
			if (p_ >= end_) {
				Bad("truncated escape");
			}
			switch (*p_) {
			case 'n':
				out += '\n';
				break;
			case 't':
				out += '\t';
				break;
			case 'r':
				out += '\r';
				break;
			case 'b':
				out += '\b';
				break;
			case 'f':
				out += '\f';
				break;
			case 'u': {
				if (end_ - p_ < 5) {
					Bad("truncated \\u escape");
				}
				const unsigned code = static_cast<unsigned>(std::strtoul(std::string(p_ + 1, 4).c_str(), nullptr, 16));
				// Names here are ASCII; anything wider is kept as the UTF-8 of the BMP code point.
				if (code < 0x80) {
					out += static_cast<char>(code);
				} else if (code < 0x800) {
					out += static_cast<char>(0xC0 | (code >> 6));
					out += static_cast<char>(0x80 | (code & 0x3F));
				} else {
					out += static_cast<char>(0xE0 | (code >> 12));
					out += static_cast<char>(0x80 | ((code >> 6) & 0x3F));
					out += static_cast<char>(0x80 | (code & 0x3F));
				}
				p_ += 4;
				break;
			}
			default:
				out += *p_; // \" \\ \/
			}
			p_++;
		}
		if (p_ >= end_) {
			Bad("unterminated string");
		}
		p_++;
		return out;
	}
	Json Value(int depth) {
		if (depth > 64) {
			Bad("nesting too deep");
		}
		Skip();
		if (p_ >= end_) {
			Bad("unexpected end");
		}
		Json v;
		if (*p_ == '{') {
			p_++;
			v.type = Json::Object;
			if (Eat('}')) {
				return v;
			}
			do {
				Skip();
				std::string key = String();
				if (!Eat(':')) {
					Bad("expected ':'");
				}
				v.object[key] = Value(depth + 1);
			} while (Eat(','));
			if (!Eat('}')) {
				Bad("expected '}'");
			}
		} else if (*p_ == '[') {
			p_++;
			v.type = Json::Array;
			if (Eat(']')) {
				return v;
			}
			do {
				v.array.push_back(Value(depth + 1));
			} while (Eat(','));
			if (!Eat(']')) {
				Bad("expected ']'");
			}
		} else if (*p_ == '"') {
			v.type = Json::String;
			v.str = String();
		} else if (Literal("true")) {
			v.type = Json::Bool;
			v.number = 1;
		} else if (Literal("false")) {
			v.type = Json::Bool;
		} else if (Literal("null")) {
			v.type = Json::Null;
		} else {
			// strtod needs a terminator; the buffer is not guaranteed to have one past end_, so copy a window.
			const std::string window(p_, static_cast<size_t>(end_ - p_ > 40 ? 40 : end_ - p_));
			char *stop = nullptr;
			const double d = std::strtod(window.c_str(), &stop);
			if (stop == window.c_str()) {
				Bad("unexpected character");
			}
			p_ += stop - window.c_str();
			v.type = Json::Number;
			v.number = d;
		}
		return v;
	}
};

std::string ReadAll(const std::string &path, const char *what) {
	std::ifstream f(path, std::ios::binary);
	if (!f) {
		Fail(std::string("cannot open ") + what + " '" + path + "'");
	}
	std::ostringstream ss;
	ss << f.rdbuf();
	return ss.str();
}

size_t DtypeSize(const std::string &dtype) {
	if (dtype == "F32") {
		return 4;
	}
	if (dtype == "F16" || dtype == "BF16") {
		return 2;
	}
	return 0;
}

} // namespace

std::map<std::string, WeightEntry> LoadWeightIndex(const std::string &tensor_map_path,
                                                   const std::string &safetensors_path) {
	// Tensor map: {"initializers": {name: key}, "transforms": {...}}
	const std::string map_text = ReadAll(tensor_map_path, "tensor map");
	const Json map = Parser(map_text.data(), map_text.size(), "tensor map '" + tensor_map_path + "'").Parse();
	const Json *inits = map.Find("initializers");
	if (!inits || inits->type != Json::Object) {
		Fail("tensor map '" + tensor_map_path + "' has no \"initializers\" object");
	}
	const Json *transforms = map.Find("transforms");
	if (transforms && transforms->type == Json::Object && !transforms->object.empty()) {
		Fail("tensor map '" + tensor_map_path + "' lists transforms (e.g. '" + transforms->object.begin()->first +
		     "') and the mlx backend does not apply them; run this model on the CPU");
	}

	// Safetensors header: [u64 LE length][JSON]
	std::ifstream f(safetensors_path, std::ios::binary);
	if (!f) {
		Fail("cannot open checkpoint '" + safetensors_path + "'");
	}
	struct stat st {};
	if (::stat(safetensors_path.c_str(), &st) != 0) {
		Fail("cannot stat checkpoint '" + safetensors_path + "'");
	}
	const size_t file_size = static_cast<size_t>(st.st_size);
	unsigned char lenbuf[8];
	f.read(reinterpret_cast<char *>(lenbuf), 8);
	if (f.gcount() != 8) {
		Fail("checkpoint '" + safetensors_path + "' is truncated (no header length)");
	}
	uint64_t header_len = 0;
	for (int i = 7; i >= 0; i--) {
		header_len = (header_len << 8) | lenbuf[i];
	}
	if (header_len == 0 || header_len > file_size - 8) {
		Fail("checkpoint '" + safetensors_path + "' has an invalid header length (" + std::to_string(header_len) +
		     " bytes in a " + std::to_string(file_size) + " byte file)");
	}
	std::string header(header_len, '\0');
	f.read(&header[0], static_cast<std::streamsize>(header_len));
	if (static_cast<uint64_t>(f.gcount()) != header_len) {
		Fail("checkpoint '" + safetensors_path + "' is truncated (header)");
	}
	const Json hdr = Parser(header.data(), header.size(), "checkpoint '" + safetensors_path + "' header").Parse();
	const size_t base = 8 + static_cast<size_t>(header_len);

	std::map<std::string, WeightEntry> out;
	for (const auto &kv : inits->object) {
		if (kv.second.type != Json::String) {
			Fail("tensor map entry '" + kv.first + "' is not a string");
		}
		const Json *t = hdr.Find(kv.second.str);
		if (!t || t->type != Json::Object) {
			Fail("checkpoint '" + safetensors_path + "' has no tensor '" + kv.second.str + "' (initializer '" +
			     kv.first + "'); is it the checkpoint this model was exported from?");
		}
		const Json *dtype = t->Find("dtype");
		const Json *shape = t->Find("shape");
		const Json *offs = t->Find("data_offsets");
		if (!dtype || dtype->type != Json::String || !shape || shape->type != Json::Array || !offs ||
		    offs->type != Json::Array || offs->array.size() != 2) {
			Fail("tensor '" + kv.second.str + "' has a malformed header entry");
		}
		WeightEntry e;
		e.dtype = dtype->str;
		const size_t esz = DtypeSize(e.dtype);
		if (!esz) {
			Fail("tensor '" + kv.second.str + "' has dtype " + e.dtype + ", the mlx backend reads F32, F16 and BF16");
		}
		size_t count = 1;
		for (const auto &d : shape->array) {
			e.shape.push_back(static_cast<int64_t>(d.number));
			count *= static_cast<size_t>(d.number);
		}
		const size_t begin = static_cast<size_t>(offs->array[0].number);
		const size_t finish = static_cast<size_t>(offs->array[1].number);
		if (finish < begin || base + finish > file_size) {
			Fail("tensor '" + kv.second.str + "' runs past the end of the checkpoint");
		}
		if (finish - begin != count * esz) {
			Fail("tensor '" + kv.second.str + "' holds " + std::to_string(finish - begin) +
			     " bytes, its shape needs " + std::to_string(count * esz));
		}
		e.offset = base + begin;
		e.length = finish - begin;
		out[kv.first] = std::move(e);
	}
	return out;
}

} // namespace mlxgraph
} // namespace anofox
