#pragma once

#include <cstdint>
#include <atomic>

namespace PLANS {

	typedef int64_t Seed;

	class Random {
		private:
			static const int64_t SEED_MULTIPLIER;
			static const int64_t SEED_MASK;
			static const int64_t SEED_ADDEND;
			static const int64_t SEED_UNIQUIFIER_MULTIPLIER;
			static std::atomic<int64_t> SEED_UNIQUIFIER_MAGIC_STATE;

			Seed seed;

			static Seed uniquifySeed(Seed seed);

			void init();
			int32_t next(uint32_t bits);
		protected:
		public:
			Random();
			explicit Random(Seed seed);
			~Random();
			void setSeed(Seed seed);
			uint32_t nextUInt(uint32_t bound = UINT32_MAX);
			int32_t nextIntInRange(int32_t minimum, int32_t maximum);
	};
}