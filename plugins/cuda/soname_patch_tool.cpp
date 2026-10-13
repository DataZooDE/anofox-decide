// soname_patch_tool: rename the SONAME of an ONNX Runtime core, in place, using the very routine the extension's
// decide_download_runtime uses on the wheel (src/include/decide_soname_patch.hpp), so the plugin is built against
// bytes patched exactly as the ones a user receives.
//
//   soname_patch_tool <libonnxruntime.so.X.Y.Z> <output.so>
//
// Exits non-zero unless exactly one SONAME string was found (zero: the distribution changed shape; several: the
// only-the-.dynstr-entry assumption broke).
#include "decide_soname_patch.hpp"

#include <cstdio>
#include <fstream>
#include <iterator>
#include <string>

int main(int argc, char **argv) {
	if (argc != 3) {
		std::fprintf(stderr, "usage: %s <core.so> <output.so>\n", argv[0]);
		return 2;
	}
	std::ifstream in(argv[1], std::ios::binary);
	if (!in) {
		std::fprintf(stderr, "cannot read %s\n", argv[1]);
		return 1;
	}
	std::string data((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
	const auto replaced = duckdb::anofox::DecidePatchOrtSonameInPlace(&data[0], data.size());
	if (replaced != 1) {
		std::fprintf(stderr, "expected exactly one SONAME string in %s, found %zu\n", argv[1], replaced);
		return 1;
	}
	std::ofstream out(argv[2], std::ios::binary | std::ios::trunc);
	out.write(data.data(), (std::streamsize)data.size());
	out.close();
	if (!out) {
		std::fprintf(stderr, "cannot write %s\n", argv[2]);
		return 1;
	}
	std::printf("renamed core written: %s\n", argv[2]);
	return 0;
}
