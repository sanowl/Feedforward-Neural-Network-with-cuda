#include "neural_network.cuh"
#include <iostream>
#include <chrono>
#include <sstream>
#include <iomanip>

#define NUM_EPOCHS 10
#define NUM_BATCHES 1000
#define NUM_VALIDATION_BATCHES 100

int main() {
    try {
        // Seed for reproducibility
        srand(1234);

        // Initialize neural network
        NeuralNetwork nn;

        // Allocate host memory for input and labels
        std::vector<float> input(BATCH_SIZE * INPUT_SIZE);
        std::vector<float> labels(BATCH_SIZE * OUTPUT_SIZE);

        // Start training
        for (int epoch = 0; epoch < NUM_EPOCHS; ++epoch) {
            float total_loss = 0.0f;
            auto start_time = std::chrono::high_resolution_clock::now();

            for (int batch = 0; batch < NUM_BATCHES; ++batch) {
                // Load batch data
                load_batch_data(input, labels, BATCH_SIZE, INPUT_SIZE, OUTPUT_SIZE);

                // Forward pass
                nn.forward(input.data());

                // Backward pass
                float loss = nn.backward(labels.data());

                // Update weights
                nn.update_weights();

                total_loss += loss;
            }

            // Adjust learning rate
            nn.adjust_learning_rate(epoch);

            // Perform validation
            float avg_val_loss = validate(nn, NUM_VALIDATION_BATCHES);

            auto end_time = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> epoch_duration = end_time - start_time;

            std::cout << "Epoch " << std::setw(2) << epoch + 1 << "/" << NUM_EPOCHS 
                      << ", Training Loss: " << std::fixed << std::setprecision(6) << total_loss / NUM_BATCHES 
                      << ", Validation Loss: " << avg_val_loss 
                      << ", Time: " << std::setprecision(2) << epoch_duration.count() << "s" << std::endl;

            // Save model after each epoch
            std::stringstream ss;
            ss << "model_epoch_" << std::setw(2) << std::setfill('0') << epoch + 1 << ".bin";
            nn.save_model(ss.str());
        }

        std::cout << "Training completed successfully." << std::endl;

        // Optional: Load and test a saved model
        std::cout << "Testing saved model..." << std::endl;
        NeuralNetwork test_nn;
        test_nn.load_model("model_epoch_10.bin");

        // Generate test data
        std::vector<float> test_input(BATCH_SIZE * INPUT_SIZE);
        std::vector<float> test_labels(BATCH_SIZE * OUTPUT_SIZE);
        load_batch_data(test_input, test_labels, BATCH_SIZE, INPUT_SIZE, OUTPUT_SIZE);

        // Forward pass with test data
        test_nn.forward(test_input.data());

        // Compute and print test loss
        float test_loss = test_nn.backward(test_labels.data(), true);  // true for validation mode
        std::cout << "Test Loss: " << std::fixed << std::setprecision(6) << test_loss << std::endl;
    }
    catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}