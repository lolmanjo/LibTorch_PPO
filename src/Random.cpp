#include "Random.h"

#include <ctime>
#include <limits>

using namespace PLANS;

const int64_t Random::SEED_MULTIPLIER = 0x5DEECE66DL;
const int64_t Random::SEED_MASK = (static_cast<int64_t>(1) << 48) - 1;
const int64_t Random::SEED_ADDEND = 0xBL;
const int64_t Random::SEED_UNIQUIFIER_MULTIPLIER = 3847689576984895681L;
std::atomic<int64_t> Random::SEED_UNIQUIFIER_MAGIC_STATE = 9348568596859952L;

Seed Random::uniquifySeed(Seed seed) {
	//Idea: Small difference in input leads to a large difference in output
	int64_t uniquifier = 0L;
	int64_t nextUniquifier = 0L;
	do {
		uniquifier = Random::SEED_UNIQUIFIER_MAGIC_STATE.load();
		nextUniquifier = uniquifier * SEED_UNIQUIFIER_MULTIPLIER;
	} while(!SEED_UNIQUIFIER_MAGIC_STATE.compare_exchange_strong(uniquifier, nextUniquifier));
	return static_cast<Seed>(nextUniquifier ^ seed);
}

void Random::init() {
	seed = (seed ^ SEED_MULTIPLIER) & SEED_MASK;
}

int32_t Random::next(uint32_t bits) {
	seed = (seed * SEED_MULTIPLIER + SEED_ADDEND) & SEED_MASK;
	return static_cast<int32_t>(static_cast<uint64_t>(seed) >> (48 - bits));
}

Random::Random() {
	seed = uniquifySeed(static_cast<Seed>(time(nullptr)));
	init();
}

Random::Random(Seed seed) : seed(seed) {
	init();
}

Random::~Random() {}

void Random::setSeed(Seed seed) {
	this->seed = seed;
	init();
}

uint32_t Random::nextUInt(uint32_t bound) {
	return next(32) % bound;
}

int32_t Random::nextIntInRange(int32_t minimum, int32_t maximum) {
	return minimum + static_cast<int32_t>(nextUInt(static_cast<uint32_t>(maximum - minimum + 1)));
}
