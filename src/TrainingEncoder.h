#pragma once

#include "TrainingController.h"

namespace PLANS {

	class Environment;

	class TrainingEncoder {
		public:
			static StateData* buildInputTensor(AGENT_ID agentID, Environment* environment);

			static float decodeAction(AGENT_ID agentID, const torch::Tensor& actorOutput);
		protected:
		private:
	};

}
