#include "TrainingController.h"

#include <filesystem>
#include <chrono>
#include <random>

#include "TrainingParameters.h"
#include "TrainingRewarder.h"
#include "TrainingEncoder.h"
#include "Environment.h"
#include "Maths.h"

using namespace PLANS;

//############################ Agent ############################

Agent::Agent(AGENT_ID agentID) : agentID(agentID), model(nullptr), states(), actions(), logProbs(), values(), rewards(), totalReward(0.0), rewardsCount(0) {
	
	// Reserve memory for rewards. 
	rewards.reserve(TrainingController::getTrainingParameters()->trainingStepLength);

	// Create model. 
	model = new Model(STD);
	//model = new Model(TrainingController::LSTM_INPUT_SIZE, TrainingController::LSTM_OUTPUT_SIZE, STD);
	model->get()->normal(0.0, STD);

	// Move model to GPU before initializing optimizers, as doing it afterwards can cause problems with the optimizers references to the models parameters. 
#ifdef USE_CUDA
	model->get()->toDevice(torch::kCUDA);
#else
	model->get()->toDevice(torch::kCPU);
#endif
	
	// Create optimizer. 
	optimizer = new Optimizer(model->get()->parameters(), torch::optim::AdamOptions(TrainingController::params->learningRate));
}

Agent::~Agent() {
	delete model;
	delete optimizer;
}

//############################ StateData ############################

StateData::StateData(uint32_t stepIndex) : stepIndex(stepIndex), inputTensor(nullptr), inputTensorDevice(nullptr) {}

StateData::~StateData() {
	//inputTensor = torch::empty(1);
	//inputTensorDevice = torch::empty(1);
}

//############################ TrainingController ############################

// Random engine for shuffling memory.
std::random_device rd;
std::mt19937 re(rd());

bool TrainingController::verbose = true;
TrainingParameters* TrainingController::params = nullptr;
Environment* TrainingController::enviroment = nullptr;
torch::TensorOptions TrainingController::tensorOptions = torch::TensorOptions();
torch::TensorOptions TrainingController::tensorOptionsCPU = torch::TensorOptions();

// Runtime variables. 
uint32_t TrainingController::trainedEpisodes;
uint32_t TrainingController::stepsInThisEpisode;
uint32_t TrainingController::stepsTillAction;
std::vector<std::vector<StateData*>> TrainingController::stateDatas;

std::mutex TrainingController::stateDataMutex = {};
std::mutex TrainingController::backwardMutex = {};

std::vector<Agent*> TrainingController::agents;

void TrainingController::initAgents(uint32_t numOfAgents) {
	if(!agents.empty()) {
		consoleOut("TrainingController::initAgents: Agents already initialized.");
		return;
	}
	// Create agents. 
	for(uint32_t i = 0; i < numOfAgents; i++) {
		agents.push_back(new Agent(i));
	}

	trainedEpisodes = 0;
}

void TrainingController::rewardAgent(Agent* agent, bool didTakeAction, Environment* enviroment) {
	// Determine reward via rewarder. 
	double reward = TrainingRewarder::calculateReward(agent->agentID, didTakeAction, enviroment);

#ifdef MEASURE_TIME_GOAL
	if(agent->agentID == 0) {
		TRAINING_STEPS++;
		if(reward >= 100.0) {
			auto start = std::chrono::time_point_cast<std::chrono::seconds>(TRAINING_START).time_since_epoch().count();
			auto finish = std::chrono::time_point_cast<std::chrono::seconds>(std::chrono::system_clock::now()).time_since_epoch().count();
			auto total = finish - start;
			std::cout << "Training ran from " + std::to_string(start) + " to " + std::to_string(finish) + "(total of " + std::to_string(total) + " seconds) in " + std::to_string(TRAINING_STEPS) + " training steps" << std::endl;
			abort();
		}
	}
#endif

	// Add reward. 
	agent->rewards.push_back(reward);
	agent->totalReward += reward;
	agent->rewardsCount++;
}

void TrainingController::resetAgentTrainingStep(Agent* agent) {
	agent->states.clear();
	agent->actions.clear();
	agent->values.clear();
	agent->logProbs.clear();
	agent->rewards.clear();
	agent->totalReward = 0.0;
	agent->rewardsCount = 0;
}

void TrainingController::optimizePPO(Agent* agent) {
	
	consoleOut("TrainingController::optimizePPO: Agent " + std::to_string(agent->agentID) + ", total reward: " + std::to_string(agent->totalReward));

	torch::Tensor t_values = torch::cat(agent->values).detach().to(torch::kCPU);

	// Calculate the returns. 
	torch::Tensor gae = torch::zeros(1, TrainingController::getTensorOptionsCPU());
	std::vector<torch::Tensor> returns = std::vector<torch::Tensor>(agent->rewards.size(), torch::zeros(1, TrainingController::getTensorOptionsCPU()));
	torch::Tensor delta;
	for(size_t i = agent->rewards.size() - 1; i > 0; i--) {
		delta = agent->rewards[i] + TrainingController::params->ppo_gamma * t_values[i - 1] - t_values[i];	// See [SchulmanEtAl, 2017]_PPO, Equation (12)
		gae = delta + TrainingController::params->ppo_gamma * TrainingController::params->ppo_lambda * gae;

		returns[i] = gae + t_values[i];
	}

	// Build copies of tensors. 
	torch::Tensor t_logProbs = torch::cat(agent->logProbs).detach().to(torch::kCPU);
	torch::Tensor t_returns = torch::cat(returns).detach();
	torch::Tensor t_states = torch::cat(agent->states).to(torch::kCPU);
	torch::Tensor t_actions = torch::cat(agent->actions).to(torch::kCPU);

	// Calculate the advantages. 
	torch::Tensor t_advantages = (t_returns - t_values).slice(0, 0, TrainingController::params->trainingStepLength);	// Size: { trainingStepLength(120) }

	int64_t a_a = t_advantages.dim();
	int64_t a_b = t_advantages.size(0);

	// NOTE: From here the sources are https://github.com/mhubii/ppo_libtorch/tree/master and https://github.com/ericyangyu/PPO-for-Beginners. 
	double beta = TrainingController::params->ppo_beta;
	torch::Tensor mini_states;
	torch::Tensor mini_actions;
	torch::Tensor mini_logProbs;
	torch::Tensor mini_returns;
	torch::Tensor mini_advantages;
	for(uint32_t i = 0; i < TrainingController::params->ppo_epochs; i++) {
		// Construct mini batch. 
		mini_states = torch::zeros({ TrainingController::params->ppo_miniBatchSize, LSTM_INPUT_SIZE }, TrainingController::getTensorOptions());
		mini_actions = torch::zeros({ TrainingController::params->ppo_miniBatchSize, LSTM_OUTPUT_SIZE }, TrainingController::getTensorOptions());
		mini_logProbs = torch::zeros({ TrainingController::params->ppo_miniBatchSize, LSTM_OUTPUT_SIZE }, TrainingController::getTensorOptions());
		mini_returns = torch::zeros({ TrainingController::params->ppo_miniBatchSize, 1 }, TrainingController::getTensorOptions());
		mini_advantages = torch::zeros({ TrainingController::params->ppo_miniBatchSize, 1 }, TrainingController::getTensorOptions());

		uint32_t idx;
		for(uint32_t b = 0; b < TrainingController::params->ppo_miniBatchSize; b++) {
			//idx = std::uniform_int_distribution<uint32_t>(0, TrainingController::params->trainingStepLength - 1)(re);	// Randomize order. 
			idx = (i * TrainingController::params->ppo_miniBatchSize) + b;
			mini_states[b] = t_states[idx];
			mini_actions[b] = t_actions[idx];
			mini_logProbs[b] = t_logProbs[idx];
			mini_returns[b] = t_returns[idx];
			mini_advantages[b] = t_advantages[idx];
		}

		mini_states.dim();
		mini_states.size(0); // trainingStepLength
		mini_states.size(1); // TrainingController::LSTM_INPUT_SIZE

		//std::vector<torch::Tensor> av = agent->model->get()->forward(mini_states, false); // action value pairs

		std::tuple<torch::Tensor, torch::Tensor> av = agent->model->get()->forward(mini_states, false); // action value pairs
		torch::Tensor action = std::get<0>(av);
		torch::Tensor entropy = agent->model->get()->entropy().mean();
		torch::Tensor new_log_prob = agent->model->get()->logProb(mini_actions);

		torch::Tensor old_log_prob = mini_logProbs;
		torch::Tensor ratio = (new_log_prob - old_log_prob).exp();
		torch::Tensor surr1 = ratio * mini_advantages;	//  ratio is { 30, 6 }, mini_advantages is { 30, 1 }. 
		torch::Tensor surr2 = torch::clamp(ratio, 1.0 - beta, 1.0 + beta) * mini_advantages;

		torch::Tensor val = std::get<1>(av);
		torch::Tensor actorLoss = -torch::min(surr1, surr2).mean();
		torch::Tensor criticLoss = (mini_returns - val).pow(2).mean();

		// Calculate total loss. 
		torch::Tensor totalLoss = 0.5 * criticLoss + actorLoss - TrainingController::params->ppo_gamma * entropy;

		// Lock mutex because of potentially asynchronous backward call
		backwardMutex.lock();

		//agent->model->get()->actor.get()->weight.dim();
		//float f_1 = agent->model->get()->actor.get()->weight[0][0].item().toFloat();
		//float f_1 = agent->model->get()->a_lin3_->weight[0][0].item().toFloat();
		float loss = totalLoss.item().toFloat();

		// Update model. 
		agent->optimizer->zero_grad();
		//torch::Tensor tt = totalLoss.grad();
		if(i != TrainingController::params->ppo_epochs - 1) {
			// Retain graph if this is not the last ppo_epoch. 
			totalLoss.backward({}, true);
		} else {
			// This is the last ppo_epoch, don't retain the graph. 
			totalLoss.backward({}, false);
		}
		//torch::Tensor tt_2 = totalLoss.grad();
		agent->optimizer->step();

		//float f_2 = agent->model->get()->actor.get()->weight[0][0].item().toFloat();
		//float f_2 = agent->model->get()->a_lin3_->weight[0][0].item().toFloat();

		// Unlock mutex. 
		backwardMutex.unlock();
	}
}

void TrainingController::consoleOut(const std::string& output) {
	if(verbose) {
		std::cout << output << std::endl;
		//LogHelper::info(output);
	}
}

bool TrainingController::init(TrainingParameters* parameters, Environment* enviroment) {
	if(params != nullptr) {
		consoleOut("TrainingController::init: Already initialized.");
		return false;
	}
	if(parameters == nullptr) {
		consoleOut("TrainingController::init: Given parameters pointer is nullptr.");
		return false;
	}
	TrainingController::params = parameters;

	if(enviroment == nullptr) {
		consoleOut("TrainingController::init: Given enviroment pointer is nullptr.");
		return false;
	}
	TrainingController::enviroment = enviroment;

	// Init tensor options. 
#ifdef USE_CUDA
	tensorOptions = tensorOptions.device(torch::kCUDA).dtype(torch::kFloat32).requires_grad(false);
#else
	tensorOptions = tensorOptions.device(torch::kCPU).dtype(torch::kFloat32).requires_grad(false);
#endif
	tensorOptionsCPU = tensorOptionsCPU.device(torch::kCPU).dtype(torch::kFloat32).requires_grad(false);

	trainedEpisodes = 0;

	// Pre-fill state datas. 
	for(uint32_t i = 0; i < NUM_OF_AGENTS; i++) {
		stateDatas.push_back(std::vector<StateData*>());
	}

	initAgents(NUM_OF_AGENTS);

	stepsInThisEpisode = -1;
	stepsTillAction = params->policyStepLength - 1;

	return true;	// Let the program proceed. 
}

bool TrainingController::onNextScenarioRequired(bool isInit) {
	consoleOut("TrainingController::onNextScenarioRequired");

	if(!isInit) {
		trainedEpisodes++;
	}

	// Check if episode limit reached. If so, terminate training. 
	if(trainedEpisodes >= params->maxEpisodes && params->maxEpisodes > 0) {
		return true;	// Terminate. Maximum episode count reached. 
	}

	// Reset environment. 
	enviroment->reset(NUM_OF_AGENTS);

	// Reset episode values. 
	stepsInThisEpisode = -1;
	stepsTillAction = params->policyStepLength - 1;
	TrainingController::cleanUpStateDatas();

	// Reset all agents. 
	for(Agent* agent : agents) {
		resetAgentTrainingStep(agent);
	}

	return false;	// Don't terminate, there are episodes to do left. 
}

bool TrainingController::onActionRequired(AGENT_ID agentID, std::vector<torch::Tensor>& output) {
	// Check wether action is allowed (policy step). 
	if(stepsTillAction != 0) {
		return false;	// Not allowed in this step. 
	}

	// Get agent. 
	Agent* agent = agents[agentID];

#if defined(_DEBUG) and defined(MEASURE_TIME)
	auto start = std::chrono::time_point_cast<std::chrono::microseconds>(std::chrono::system_clock::now()).time_since_epoch().count();
#endif

	// Get or create state data. 
	const StateData* stateData = TrainingController::getOrCreateStateData(agentID);

	// Save input tensor (state tensor). 
	agent->states.push_back(stateData->inputTensor);

	// Pass inputs into model to produce actor and critic outputs. 
	std::tuple<torch::Tensor, torch::Tensor> outputTuple = agent->model->get()->forward(stateData->inputTensorDevice, true);

#if defined(_DEBUG) and defined(MEASURE_TIME)
	auto finish = std::chrono::time_point_cast<std::chrono::microseconds>(std::chrono::system_clock::now()).time_since_epoch().count();
	auto total = finish - start;
	std::cout << "[" + std::to_string(agentID) + "] ran from " + std::to_string(start) + " to " + std::to_string(finish) + "(total of " + std::to_string(total) + ")" << std::endl;
#endif

	// Extract actual output (vector index, sequence, batch). 
	torch::Tensor actorOutput = std::get<0>(outputTuple)[0];
	torch::Tensor criticOutput = std::get<1>(outputTuple)[0];

	// Save action returned by the actor (just the action type at index 0). 
	agent->actions.push_back(torch::full(1, actorOutput[0].item()));

	// Create and save logProb. 
	torch::Tensor logProb = agent->model->get()->logProb(actorOutput[0]);
	agent->logProbs.push_back(logProb);

	// Save value returned by the critic. 
	agent->values.push_back(criticOutput);

	// Copy output tensor to CPU for faster access while decoding. 
	actorOutput = actorOutput.to(torch::kCPU);

	// Process outputs to action. 
	uint32_t delay = 0;
	float action = TrainingEncoder::decodeAction(agent->agentID, actorOutput);

	// Reward agent. Accumulate rewards from occured events, which will be deleted now. 
	TrainingController::rewardAgent(agent, action != UINT8_MAX, enviroment);

	// Fill output vector. 
	output.push_back(actorOutput);
	output.push_back(criticOutput);

	return true;
}

void TrainingController::onAgentExecuted(AGENT_ID agentID) {
	// Get agent. 
	Agent* agent = agents[agentID];

	if(agent->rewards.size() < TrainingController::params->trainingStepLength) {
		return;	// Not enough rewards. 
	}

	if(agent->totalReward != 0.0) {

#ifdef _DEBUG
		//float f_1 = agent->model->get()->actor_0->weight[0][0].item().toFloat();

		uint32_t tmpSize = static_cast<uint32_t>(agent->model->get()->actor_0->weight.size(1));

		std::vector<float> befores = std::vector<float>();
		befores.reserve(tmpSize);
		for(uint32_t i = 0; i < tmpSize; i++) {
			befores.push_back(agent->model->get()->actor_0->weight[0][i].item().toFloat());
		}
#endif

		// Optimize the agent based on the PPO algorithm. 
		optimizePPO(agent);

#ifdef _DEBUG
		//float f_2 = agent->model->get()->actor_0->weight[0][0].item().toFloat();

		std::vector<float> afters = std::vector<float>();
		afters.reserve(tmpSize);
		for(uint32_t i = 0; i < tmpSize; i++) {
			afters.push_back(agent->model->get()->actor_0->weight[0][i].item().toFloat());
		}

		std::vector<float> differences = std::vector<float>();
		differences.reserve(tmpSize);
		for(uint32_t i = 0; i < tmpSize; i++) {
			differences.push_back(Maths::abs(befores[i] - afters[i]));
		}
		bool anyDifferences = false;
		for(uint32_t i = 0; i < tmpSize; i++) {
			if(!Maths::compareFloat(differences[i], 0.0F)) {
				anyDifferences = true;
			}
		}
		if(!anyDifferences) {
			consoleOut("TrainingController::onAgentExecuted: No differences detected.");
		}
#endif

	} else {
		consoleOut("TrainingController::onAgentExecuted: Not optimizing agent " + std::to_string(agent->agentID) + ", because his total reward is 0.0.");
	}

	// Count actions. 
	std::vector<uint32_t> actionCounts = std::vector<uint32_t>(5);
	for(uint32_t i = 0; i < agent->actions.size(); i++) {
		uint32_t action = static_cast<uint32_t>(agent->actions[i].item().toInt());
		if(action < 4) {
			actionCounts[action]++;
		} else {
			actionCounts[4]++;
		}
	}

	// Reset agent. 
	resetAgentTrainingStep(agent);

	// Clean up state datas. 
	TrainingController::cleanUpStateDatas();

	//// Clear GPU cache still hold by libtorch. 
	//c10::cuda::CUDACachingAllocator::emptyCache();
}

bool TrainingController::onGameTickPassed() {
	// Update stepsTillAction. 
	if(params->policyStepLength > 1) {
		if(stepsTillAction == 0 && stepsInThisEpisode > 0) {
			stepsTillAction = params->policyStepLength - 1;
		} else {
			stepsTillAction--;
		}
	}
	// Increase step counter in this episode. 
	stepsInThisEpisode++;
	// Check if episode reached maximum length. 
	bool episodeReachedMaxLength = stepsInThisEpisode >= params->maxEpisodeLength;
	
	if(episodeReachedMaxLength) {
		return true;	// Terminate episode. 
	}
	return false;	// False = no need to terminate episode. 
}

void TrainingController::cleanUp() {
	// Clean up parameters. 
	if(params != nullptr) {
		delete params;
		params = nullptr;
	}
	// Clean up state datas. 
	TrainingController::cleanUpStateDatas();
	// Clean up agents. 
	for(Agent* agent : agents) {
		delete agent;
	}
	agents.clear();
}

bool TrainingController::isVerbose() {
	return verbose;
}

const TrainingParameters* TrainingController::getTrainingParameters() {
	return params;
}

const torch::TensorOptions& TrainingController::getTensorOptions() {
	return tensorOptions;
}

const torch::TensorOptions& TrainingController::getTensorOptionsCPU() {
	return tensorOptionsCPU;
}

uint32_t TrainingController::getStepsInThisEpisode() {
	return stepsInThisEpisode;
}

void TrainingController::addStateData(AGENT_ID agentID, const StateData* stateData) {
	stateDataMutex.lock();
	std::vector<StateData*>& list = stateDatas[agentID];
#ifdef _DEBUG
	const StateData* tmp = TrainingController::getStateData(agentID, stateData->stepIndex);
	if(tmp != nullptr) {
		consoleOut("TrainingController::addStateData: LinkedList already contains a state data with step index \"" + std::to_string(stateData->stepIndex) + "\"");
		stateDataMutex.unlock();
		return;
	}
#endif
	list.push_back(const_cast<StateData*>(stateData));
	stateDataMutex.unlock();
}

const StateData* TrainingController::getStateData(AGENT_ID agentID, uint32_t stepIndex) {
	std::vector<StateData*>& list = stateDatas[agentID];
	for(const StateData* stateData : list) {
		if(stateData->stepIndex == stepIndex) {
			return stateData;
		}
	}
	return nullptr;
}

const StateData* TrainingController::getOrCreateStateData(AGENT_ID agentID) {
	// Get or create state data. 
	stateDataMutex.lock();
	const StateData* stateData = TrainingController::getStateData(agentID, stepsInThisEpisode);
	if(stateData != nullptr) {
		stateDataMutex.unlock();
		return stateData;
	} else {
		// Build state data via encoder. 
		stateData = TrainingEncoder::buildInputTensor(agentID, enviroment);
		// Save state data, as it's new. 
		stateDataMutex.unlock();
		TrainingController::addStateData(agentID, stateData);
		return stateData;
	}
}

void TrainingController::cleanUpStateDatas() {
	stateDataMutex.lock();
	for(std::vector<StateData*>& list : stateDatas) {
		for(StateData* stateData : list) {
			delete stateData;
		}
		list.clear();
	}
	stateDataMutex.unlock();
}
