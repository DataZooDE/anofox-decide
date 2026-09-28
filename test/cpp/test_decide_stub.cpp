#include "catch.hpp"
#include "decide_provider.hpp"

using namespace duckdb::anofox;

TEST_CASE("stub provider returns deterministic 0.5 with identity", "[anofox_decide]") {
	auto r = DecideStubScore("The customer requests a refund.", "A refund is requested.", "stub");
	REQUIRE(r.probability == 0.5);
	REQUIRE(r.model == "stub");
	REQUIRE(r.provider == "stub");
}
