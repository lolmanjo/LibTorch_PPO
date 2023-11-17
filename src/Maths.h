#pragma once

namespace PLANS {

	class Maths {
		public:
			// Absolute of given float. 
			static float abs(float f) {
				return f < 0.0F ? -f : f;
			}

			static bool compareFloat(float a, float b, float epsilon = FLT_MIN) {
				return Maths::abs(a - b) < epsilon;
			}
		protected:
		private:
	};

}
