#include "Environment.h"

#include "Maths.h"

using namespace PLANS;

//############################ EnvironmentBinary ############################

EnvironmentBinary::EnvironmentBinary() : Environment(), random(), state(), actions() {}

void EnvironmentBinary::reset(uint32_t numOfAgents) {
	actions.clear();
	for(uint32_t i = 0; i < numOfAgents; i++) {
		actions.push_back(0);
	}
	// Set initial state. 
	update();
}

void EnvironmentBinary::update() {
	state = static_cast<uint8_t>(random.nextIntInRange(1, 2));
}

void EnvironmentBinary::getInputData(AGENT_ID agentID, std::vector<float>& data) {
	// Put relevant environment data into "data" via "putIntoData". 
	int currIndex = 0;
	for(int64_t i = 0; i < LSTM_INPUT_SIZE; i++) {
		putIntoData(data, currIndex, state);
	}
}

void EnvironmentBinary::onAction(AGENT_ID agentID, float action) {
	actions[agentID] = static_cast<uint8_t>(action);
}

float EnvironmentBinary::rewardAgent(AGENT_ID agentID) {
	if(state == actions[agentID]) {
		return 10.0F;
	} else {
		return -10.0F;
	}
}

//############################ EnvironmentFloat ############################

EnvironmentFloat::EnvironmentFloat() : Environment(), state(1.0F), actions() {}

void EnvironmentFloat::reset(uint32_t numOfAgents) {
	actions.clear();
	for(uint32_t i = 0; i < numOfAgents; i++) {
		actions.push_back(0.0F);
	}
}

void EnvironmentFloat::update() {}

void EnvironmentFloat::getInputData(AGENT_ID agentID, std::vector<float>& data) {
	// Put relevant environment data into "data" via "putIntoData". 
	int currIndex = 0;
	for(int64_t i = 0; i < LSTM_INPUT_SIZE; i++) {
		putIntoData(data, currIndex, state);
	}
}

void EnvironmentFloat::onAction(AGENT_ID agentID, float action) {
	actions[agentID] = action;
}

float EnvironmentFloat::rewardAgent(AGENT_ID agentID) {
	return Maths::abs(state - actions[agentID]);
}
