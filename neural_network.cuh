#ifndef NEURAL_NETWORK_CUH
#define NEURAL_NETWORK_CUH

#include <cuda_runtime.h>
#include <curand.h>
#include <cublas_v2.h>
#include <memory>
#include <string>
#include <vector>

// Define constants
#define INPUT_SIZE 784
#define HIDDEN_SIZE 256
#define OUTPUT_SIZE 10
#define BATCH_SIZE 32
#define INITIAL_LEARNING_RATE 0.01f
#define CLIP_THRESHOLD 5.0f

// Error checking macros
#define CHECK_CUDA(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA error in " << __FILE__ << ":" << __LINE__ \
                      << ", code: " << err << ", reason: " \
                      << cudaGetErrorString(err) << std::endl; \
            std::exit(EXIT_FAILURE); \
        } \
    } while(0)

#define CHECK_CUBLAS(call) \
    do { \
        cublasStatus_t status = call; \
        if (status != CUBLAS_STATUS_SUCCESS) { \
            std::cerr << "CUBLAS error in " << __FILE__ << ":" << __LINE__ \
                      << ", code: " << status << std::endl; \
            std::exit(EXIT_FAILURE); \
        } \
    } while(0)

#define CHECK_CURAND(call) \
    do { \
        curandStatus_t status = call; \
        if (status != CURAND_STATUS_SUCCESS) { \
            std::cerr << "CURAND error in " << __FILE__ << ":" << __LINE__ \
                      << ", code: " << status << std::endl; \
            std::exit(EXIT_FAILURE); \
        } \
    } while(0)

// Custom CUDA deleter for smart pointers
struct cuda_deleter {
    void operator()(float* ptr) const;
};

// Helper function to allocate device memory
template<typename T>
std::unique_ptr<T, cuda_deleter> cuda_malloc(size_t size);

class NeuralNetwork {
private:
    // Device pointers
    std::unique_ptr<float, cuda_deleter> d_input, d_hidden, d_output, d_W1, d_W2, d_b1, d_b2;
    std::unique_ptr<float, cuda_deleter> d_dW1, d_dW2, d_db1, d_db2, d_labels, d_loss;
    std::unique_ptr<float, cuda_deleter> d_d_logits, d_hidden_derivative;

    cublasHandle_t cublas_handle;
    curandGenerator_t curand_gen;

    float learning_rate;

    void initialize_weights();

public:
    cudaStream_t train_stream;
    cudaStream_t val_stream;

    NeuralNetwork(float initial_lr = INITIAL_LEARNING_RATE);
    ~NeuralNetwork();

    void forward(const float* input, bool is_validation = false);
    float backward(const float* labels, bool is_validation = false);
    void update_weights();
    void save_model(const std::string& filename);
    void load_model(const std::string& filename);
    void adjust_learning_rate(int epoch, float decay_factor = 0.95f);

    // Getters for validation
    float* get_d_loss() const { return d_loss.get(); }
    float* get_d_output() const { return d_output.get(); }
    float* get_d_labels() const { return d_labels.get(); }
    float* get_d_d_logits() const { return d_d_logits.get(); }
};

// CUDA kernel declarations
__global__ void fused_bias_relu(float* data, const float* bias, int batch_size, int size_per_batch);
__global__ void softmax_cross_entropy(const float* logits, const float* labels, float* loss, float* d_logits, int batch_size, int output_size);
__global__ void sum_gradients(const float* gradients, float* bias_grad, int batch_size, int size_per_batch);
__global__ void gradient_clipping(float* gradients, int size, float threshold);

// Helper functions
void generate_one_hot_labels(std::vector<float>& labels, int batch_size, int output_size);
void load_batch_data(std::vector<float>& input, std::vector<float>& labels, int batch_size, int input_size, int output_size);
float validate(NeuralNetwork& nn, int num_validation_batches);

#endif // NEURAL_NETWORK_CUH