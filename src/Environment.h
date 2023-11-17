#pragma once

#include "TrainingConsts.h"
#include "Random.h"

namespace PLANS {

	class Environment {
		public:
			Environment() = default;

			virtual void reset(uint32_t numOfAgents) = 0;
			virtual void update() = 0;
			virtual void getInputData(AGENT_ID agentID, std::vector<float>& data) = 0;
			virtual void onAction(AGENT_ID agentID, float action) = 0;
			virtual float rewardAgent(AGENT_ID agentID) = 0;
		protected:
			template<typename T>
			static void putIntoData(std::vector<float>& dataVector, int& currentIndex, T value) {
				dataVector[currentIndex++] = static_cast<float>(value);
			}
		private:
	};

	//############################ EnvironmentBinary ############################

	class EnvironmentBinary : public Environment {
		public:
			EnvironmentBinary();

			virtual void reset(uint32_t numOfAgents) final override;
			virtual void update() final override;
			virtual void getInputData(AGENT_ID agentID, std::vector<float>& data) final override;
			virtual void onAction(AGENT_ID agentID, float action) final override;
			virtual float rewardAgent(AGENT_ID agentID) final override;
		protected:
		private:
			Random random;
			uint8_t state;
			std::vector<uint8_t> actions;
	};

	//############################ EnvironmentFloat ############################

	class EnvironmentFloat : public Environment {
		public:
			EnvironmentFloat();

			virtual void reset(uint32_t numOfAgents) final override;
			virtual void update() final override;
			virtual void getInputData(AGENT_ID agentID, std::vector<float>& data) final override;
			virtual void onAction(AGENT_ID agentID, float action) final override;
			virtual float rewardAgent(AGENT_ID agentID) final override;
		protected:
		private:
			float state;
			std::vector<float> actions;
	};

}
