#pragma once

#pragma warning(push)
#pragma warning(disable : 4067)
#pragma warning(disable : 4275)
#pragma warning(disable : 4251)
#pragma warning(disable : 4244)
#pragma warning(disable : 4267)
#pragma warning(disable : 4624)
#include <torch/torch.h>
#pragma warning(pop)

#undef _DEBUG	// PyTorch headers seem to define _DEBUG even in release configuration. 

#ifdef _DEBUG_	// Use custom _DEBUG_ to define _DEBUG if actually intended. 
#define _DEBUG
#endif

#include <cstdint>
#include <chrono>

#define MEASURE_TIME 1
#define MEASURE_TIME_GOAL 1

namespace PLANS {

	using AGENT_ID = uint32_t;

	constexpr uint32_t NUM_OF_AGENTS = 1;

	// Version 1. 
	//static const int64_t LSTM_INPUT_SIZE = 2488;	// 2485, increased to next multiple of 8. 
	//static const int64_t LSTM_HIDDEN_SIZE = 256;

	//// Version 2. 
	//static const int64_t LSTM_INPUT_SIZE = 800;		// 795, increased to next multiple of 8. 
	//static const int64_t LSTM_HIDDEN_SIZE = 256;

	// Test. 
	static const int64_t LSTM_INPUT_SIZE = 8;		// 795, increased to next multiple of 8. 
	static const int64_t LSTM_HIDDEN_SIZE = 8;

	static const int64_t LSTM_OUTPUT_SIZE = 1;	// 6, increased to next multiple of 8. Also known as "projection size". If set to 0, TrainingController::LSTM_HIDDEN_SIZE is being used instead. 
	static const int64_t LSTM_NUM_LAYERS = 1;
	static const int64_t LSTM_SEQUENCE_LENGTH = 16;
	// In LSTM context: How many samples / rewards are collected before a weight update. As the samples are triggered by the game loop, always 1. 
	static const int64_t LSTM_BATCH_SIZE = 1;

	static const double STD = 0.02;

	static const std::chrono::system_clock::time_point TRAINING_START = std::chrono::system_clock::now();
	static int TRAINING_STEPS = 0;

}
