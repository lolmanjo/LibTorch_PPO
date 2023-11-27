#pragma once

#include <cstdint>

#include "Models.h"

namespace PLANS {

	// Determine which model to use. 
	using Model = ActorCritic;
	using ModelImpl = ActorCriticImpl;
	//using Model = ActorCriticOpenAIFive;
	//using ModelImpl = ActorCriticOpenAIFiveImpl;

	// Determine which optimizer to use. 
	using Optimizer = torch::optim::Adam;

	//############################ Agent ############################
	
	class Agent {
		public:
			Agent(AGENT_ID agentID);
			~Agent();
		protected:
		private:
			AGENT_ID agentID;

			Model* model;
			Optimizer* optimizer;

			std::vector<torch::Tensor> states;		// Tensors of size { LSTM_INPUT_SIZE }. 
			std::vector<torch::Tensor> actions;		// Tensors of size { 1 }. 
			std::vector<torch::Tensor> logProbs;
			std::vector<torch::Tensor> values;
			std::vector<double> rewards;
			double totalReward;
			uint32_t rewardsCount;

			friend class TrainingController;
	};

	//############################ StateData ############################

	struct StateData {
		uint32_t stepIndex;
		torch::Tensor inputTensor;			// Build and stays on the CPU. Used by the loss function (which runs on the CPU). 
		torch::Tensor inputTensorDevice;	// Copied to the actual device (CPU or CUDA). Used by the torch modules. 

		StateData(uint32_t stepIndex);
		~StateData();
	};
	
	//############################ TrainingController ############################

	struct TrainingParameters;
	class Environment;
	class Savegame;
	class LockStepAction;

	class TrainingController {
		public:
			static void consoleOut(const std::string& output);

			// Called from TrainMain.cpp on application start. 
			static bool init(TrainingParameters* parameters, Environment* enviroment);

			// Called from LockStepManagerTrainer, when TrainingController::onGameTickPassed returned true before. 
			// Creates a new world and activates it in the game state. Also recreates the lock step manager (LockStepManagerTraining). 
			// Returns whether the maximum episode count has been reached. 
			static bool onNextScenarioRequired(bool isInit);

			// Called when an action is required. Returns whether an action has been produced. 
			// Creates a state tensor for the current step index in the perspective of the given faction. 
			static bool onActionRequired(AGENT_ID agentID, std::vector<torch::Tensor>& output);

			// Called from NPC::executeInternal (when the npc sucessfully executed and the events will be deleted next). 
			static void onAgentExecuted(AGENT_ID agentID);

			// Called after LockStepManagerTrainer::update from StateGame::update. 
			// Indicates, that the reward for the passed game tick can be calculated. Returns whether the episode should be terminated. 
			static bool onGameTickPassed();

			// Called from TrainMain.cpp. 
			static void cleanUp();

			static bool isVerbose();
			static const TrainingParameters* getTrainingParameters();
			static const torch::TensorOptions& getTensorOptions();
			static const torch::TensorOptions& getTensorOptionsCPU();
			static uint32_t getStepsInThisEpisode();

			static void addStateData(AGENT_ID agentID, const StateData* stateData);
			static const StateData* getStateData(AGENT_ID agentID, uint32_t stepIndex);
			static const StateData* getOrCreateStateData(AGENT_ID agentID);
			static void cleanUpStateDatas();
		protected:
		private:
			static bool verbose;
			static TrainingParameters* params;
			static Environment* enviroment;
			static torch::TensorOptions tensorOptions;
			static torch::TensorOptions tensorOptionsCPU;

			static uint32_t trainedEpisodes;
			static uint32_t stepsInThisEpisode;
			static uint32_t stepsTillAction;
			static std::vector<std::vector<StateData*>> stateDatas;

			static std::mutex stateDataMutex;
			static std::mutex backwardMutex;

			static std::vector<Agent*> agents;

			static void initAgents(uint32_t numOfAgents);

			static void rewardAgent(Agent* agent, bool didTakeAction, Environment* enviroment);
			// Called after every optimizer step to reset the rewards and values of the agents. 
			static void resetAgentTrainingStep(Agent* agent);

			// Optimizes the given agent based on the PPO algorithm. 
			static void optimizePPO(Agent* agent);

			friend class Agent;
	};

}
