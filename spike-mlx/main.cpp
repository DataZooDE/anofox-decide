// Spike harness: run an ONNX graph on MLX with weights from safetensors; time it; write scores.
#include "tabfm_mlx_graph.hpp"
#include <chrono>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iterator>
using namespace anofox::mlxgraph;
using clk = std::chrono::steady_clock;
static double ms(clk::time_point a, clk::time_point b) { return std::chrono::duration<double, std::milli>(b - a).count(); }
int main(int argc, char **argv) {
	if (argc < 10) { fprintf(stderr, "usage: graph.onnx weights.safetensors index.tsv B T M inputs.bin out.bin reps\n"); return 2; }
	size_t B = atoi(argv[4]), T = atoi(argv[5]), M = atoi(argv[6]); int reps = atoi(argv[9]);
	std::vector<char> raw; { std::ifstream f(argv[7], std::ios::binary); raw.assign(std::istreambuf_iterator<char>(f), {}); }
	size_t o = 0;
	auto mk = [&](int dtype, std::vector<int> shape, size_t n, size_t esz) { Tensor t; t.shape = shape; t.dtype = dtype;
		if (dtype == 1) { t.i64.resize(n); memcpy(t.i64.data(), raw.data()+o, n*esz); } else { t.u8.resize(n); memcpy(t.u8.data(), raw.data()+o, n*esz); } o += n*esz; return t; };
	std::vector<std::pair<std::string, Tensor>> feeds;
	feeds.push_back({"input_ids", mk(1, {(int)B,(int)T}, B*T, 8)});
	feeds.push_back({"attention_mask", mk(1, {(int)B,(int)T}, B*T, 8)});
	feeds.push_back({"marker_pos", mk(1, {(int)B,(int)M}, B*M, 8)});
	feeds.push_back({"marker_mask", mk(2, {(int)B,(int)M}, B*M, 1)});
	feeds.push_back({"qtype", mk(1, {(int)B}, B, 8)});
	try {
		auto t0 = clk::now();
		Graph g(argv[1], argv[2], argv[3]);
		auto miss = g.MissingOps(); for (auto &m : miss) printf("MISSING OP %s\n", m.c_str());
		if (!miss.empty()) return 3;
		auto t1 = clk::now(); printf("load %.0f ms\n", ms(t0, t1));
		Tensor r; std::vector<double> t;
		for (int i = 0; i < reps; i++) { auto a = clk::now(); r = g.Run(feeds); auto b = clk::now(); t.push_back(ms(a, b)); }
		printf("run ms:"); for (double x : t) printf(" %.1f", x); printf("\n");
		std::ofstream f(argv[8], std::ios::binary); f.write((const char*)r.data.data(), r.data.size()*4);
		printf("out elems %zu\n", r.data.size());
	} catch (const std::exception &e) { printf("ERROR: %s\n", e.what()); return 1; }
}
