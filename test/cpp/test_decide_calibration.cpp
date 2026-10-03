#include "catch.hpp"
#include "decide_calibration.hpp"

#include <cmath>

using namespace duckdb;
using namespace duckdb::anofox;
using Catch::Matchers::Contains;

namespace {

// Deterministic uniform [0,1) (LCG), so the fit tests never flake.
struct Lcg {
	uint64_t s;
	explicit Lcg(uint64_t seed) : s(seed) {
	}
	double Next() {
		s = s * 6364136223846793005ULL + 1442695040888963407ULL;
		return (double)(s >> 11) / (double)(1ULL << 53);
	}
};

double Sigmoid(double z) {
	return 1.0 / (1.0 + std::exp(-z));
}

} // namespace

TEST_CASE("platt specs parse strictly", "[anofox_decide][calibration]") {
	DecidePlatt c;
	string why;
	SECTION("valid") {
		REQUIRE(DecideParsePlatt("platt:1.5,-0.25", c, why));
		REQUIRE(c.set);
		REQUIRE(c.a == Approx(1.5));
		REQUIRE(c.b == Approx(-0.25));
		REQUIRE(DecideParsePlatt("  platt: 2 , 0  ", c, why));
		REQUIRE(c.a == Approx(2.0));
		REQUIRE(DecideParsePlatt("platt:1e-1,3e0", c, why));
		REQUIRE(c.a == Approx(0.1));
		REQUIRE(c.b == Approx(3.0));
	}
	SECTION("rejects the wrong method and malformed numbers") {
		REQUIRE_FALSE(DecideParsePlatt("isotonic:1,2", c, why));
		REQUIRE_THAT(why, Contains("not a calibration spec"));
		REQUIRE_FALSE(DecideParsePlatt("platt:1", c, why));
		REQUIRE_THAT(why, Contains("two numbers"));
		REQUIRE_FALSE(DecideParsePlatt("platt:1,abc", c, why));
		REQUIRE_FALSE(DecideParsePlatt("platt:,2", c, why));
		REQUIRE_FALSE(DecideParsePlatt("platt:1,2x", c, why));
		REQUIRE_FALSE(c.set);
	}
	SECTION("the slope must be positive and everything finite") {
		REQUIRE_FALSE(DecideParsePlatt("platt:0,1", c, why));
		REQUIRE_THAT(why, Contains("greater than 0"));
		REQUIRE_FALSE(DecideParsePlatt("platt:-0.5,1", c, why));
		REQUIRE_THAT(why, Contains("invert"));
		REQUIRE_FALSE(DecideParsePlatt("platt:nan,1", c, why));
		REQUIRE_THAT(why, Contains("non-finite"));
		REQUIRE_FALSE(DecideParsePlatt("platt:1,inf", c, why));
	}
	SECTION("format round-trips") {
		DecidePlatt back;
		REQUIRE(DecideParsePlatt(DecideFormatPlatt(1.917975123456, -0.722736654321), back, why));
		REQUIRE(back.a == Approx(1.917975123456).epsilon(1e-8));
		REQUIRE(back.b == Approx(-0.722736654321).epsilon(1e-8));
	}
}

TEST_CASE("platt scaling maps probabilities", "[anofox_decide][calibration]") {
	DecidePlatt id;
	string why;
	REQUIRE(DecideParsePlatt("platt:1,0", id, why));
	for (double p : {0.01, 0.2, 0.5, 0.77, 0.99}) {
		REQUIRE(DecidePlattApply(id, p) == Approx(p).epsilon(1e-9));
	}
	DecidePlatt unset;
	REQUIRE(DecidePlattApply(unset, 0.123) == 0.123);

	DecidePlatt shift;
	REQUIRE(DecideParsePlatt("platt:1,-2", shift, why));
	// logit(0.5) = 0, so the output is sigmoid(b).
	REQUIRE(DecidePlattApply(shift, 0.5) == Approx(Sigmoid(-2.0)));
	// A larger slope sharpens, a smaller one flattens, both keep the ranking.
	DecidePlatt sharp, flat;
	REQUIRE(DecideParsePlatt("platt:3,0", sharp, why));
	REQUIRE(DecideParsePlatt("platt:0.3,0", flat, why));
	REQUIRE(DecidePlattApply(sharp, 0.8) > 0.8);
	REQUIRE(DecidePlattApply(flat, 0.8) < 0.8);
	REQUIRE(DecidePlattApply(sharp, 0.4) < DecidePlattApply(sharp, 0.6));
	// Exactly 0 and 1 are clamped, never infinite or NaN.
	REQUIRE(std::isfinite(DecidePlattApply(shift, 0.0)));
	REQUIRE(std::isfinite(DecidePlattApply(shift, 1.0)));
	REQUIRE(DecidePlattApply(shift, 1.0) <= 1.0);
	REQUIRE(DecidePlattApply(shift, 0.0) >= 0.0);
}

TEST_CASE("platt fit recovers a known miscalibration", "[anofox_decide][calibration]") {
	// The model reports sigmoid(f), the truth is sigmoid(2 f - 1): under-confident and biased.
	Lcg rng(12345);
	vector<double> p;
	vector<uint8_t> y;
	for (int i = 0; i < 6000; i++) {
		const double f = (rng.Next() - 0.5) * 8.0;
		p.push_back(Sigmoid(f));
		y.push_back(rng.Next() < Sigmoid(2.0 * f - 1.0) ? 1 : 0);
	}
	double a = 0, b = 0;
	string why;
	REQUIRE(DecideFitPlatt(p, y, a, b, why));
	REQUIRE(a == Approx(2.0).margin(0.15));
	REQUIRE(b == Approx(-1.0).margin(0.15));

	SECTION("applying the fit lowers the log loss") {
		DecidePlatt c;
		c.set = true;
		c.a = a;
		c.b = b;
		double before = 0, after = 0;
		for (size_t i = 0; i < p.size(); i++) {
			const double q0 = std::min(std::max(p[i], 1e-6), 1 - 1e-6);
			const double q1 = DecidePlattApply(c, p[i]);
			before -= y[i] ? std::log(q0) : std::log(1 - q0);
			after -= y[i] ? std::log(q1) : std::log(1 - q1);
		}
		REQUIRE(after < before);
	}
	SECTION("an already calibrated model fits close to the identity") {
		vector<double> p2;
		vector<uint8_t> y2;
		for (int i = 0; i < 6000; i++) {
			const double f = (rng.Next() - 0.5) * 8.0;
			p2.push_back(Sigmoid(f));
			y2.push_back(rng.Next() < Sigmoid(f) ? 1 : 0);
		}
		double a2, b2;
		REQUIRE(DecideFitPlatt(p2, y2, a2, b2, why));
		REQUIRE(a2 == Approx(1.0).margin(0.12));
		REQUIRE(b2 == Approx(0.0).margin(0.12));
	}
}

TEST_CASE("platt fit stays finite and refuses what it cannot fit", "[anofox_decide][calibration]") {
	double a = 0, b = 0;
	string why;
	SECTION("perfectly separable data gives finite coefficients (Platt's smoothed targets)") {
		vector<double> p = {0.05, 0.1, 0.2, 0.3, 0.7, 0.8, 0.9, 0.95};
		vector<uint8_t> y = {0, 0, 0, 0, 1, 1, 1, 1};
		REQUIRE(DecideFitPlatt(p, y, a, b, why));
		REQUIRE(std::isfinite(a));
		REQUIRE(std::isfinite(b));
		REQUIRE(a > 0);
	}
	SECTION("one outcome class only") {
		REQUIRE_FALSE(DecideFitPlatt({0.2, 0.4, 0.6}, {1, 1, 1}, a, b, why));
		REQUIRE_THAT(why, Contains("same outcome (true)"));
		REQUIRE_FALSE(DecideFitPlatt({0.2, 0.4, 0.6}, {0, 0, 0}, a, b, why));
		REQUIRE_THAT(why, Contains("same outcome (false)"));
	}
	SECTION("identical probabilities") {
		REQUIRE_FALSE(DecideFitPlatt({0.5, 0.5, 0.5, 0.5}, {0, 1, 0, 1}, a, b, why));
		REQUIRE_THAT(why, Contains("identical"));
	}
	SECTION("a model that ranks backwards cannot be repaired") {
		vector<double> p = {0.9, 0.8, 0.7, 0.2, 0.1, 0.3};
		vector<uint8_t> y = {0, 0, 0, 1, 1, 1};
		REQUIRE_FALSE(DecideFitPlatt(p, y, a, b, why));
		REQUIRE_THAT(why, Contains("not positive"));
	}
	SECTION("probabilities of exactly 0 and 1 do not break the fit") {
		vector<double> p = {0.0, 0.0, 0.3, 0.6, 1.0, 1.0};
		vector<uint8_t> y = {0, 0, 0, 1, 1, 1};
		REQUIRE(DecideFitPlatt(p, y, a, b, why));
		REQUIRE(std::isfinite(a));
	}
}
