#include "neural_network.cuh"
#include <iostream>
#include <fstream>
#include <cmath>
#include <algorithm>
#include <cassert>

// Implement CUDA kernels
__global__ void fused_bias_relu(float* data, const float* bias, int batch_size, int size_per_batch) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_size = batch_size * size_per_batch;
    if (idx < total_size) {
        int neuron = idx % size_per_batch;
        float value = data[idx] + bias[neuron];
        data[idx] = fmaxf(0.0f, value);
    }
}

__global__ void softmax_cross_entropy(const float* logits, const float* labels, float* loss, float* d_logits, int batch_size, int output_size) {
    extern __shared__ float shared_data[];
    int batch = blockIdx.x;
    if (batch >= batch_size) return;

    float max_val = -FLT_MAX;
    for (int i = 0; i < output_size; ++i) {
        float val = logits[batch * output_size + i];
        max_val = fmaxf(max_val, val);
    }

    float sum = 0.0f;
    for (int i = 0; i < output_size; ++i) {
        sum += expf(logits[batch * output_size + i] - max_val);
    }

    float log_sum = logf(sum) + max_val;

    float batch_loss = 0.0f;
    for (int i = 0; i < output_size; ++i) {
        int index = batch * output_size + i;
        float prob = expf(logits[index] - log_sum);
        batch_loss -= labels[index] * logf(prob + 1e-8f);
        d_logits[index] = prob - labels[index];
    }

    atomicAdd(loss, batch_loss);
}

__global__ void sum_gradients(const float* gradients, float* bias_grad, int batch_size, int size_per_batch) {
    int neuron = blockIdx.x * blockDim.x + threadIdx.x;
    if (neuron >= size_per_batch) return;

    float sum = 0.0f;
    for (int i = 0; i < batch_size; ++i) {
        sum += gradients[i * size_per_batch + neuron];
    }
    bias_grad[neuron] = sum;
}

__global__ void gradient_clipping(float* gradients, int size, float threshold) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float mag = fabsf(gradients[idx]);
        if (mag > threshold) {
            gradients[idx] = (gradients[idx] / mag) * threshold;
        }
    }
}

// Implement cuda_deleter
void cuda_deleter::operator()(float* ptr) const {
    if (ptr) {
        cudaFree(ptr);
    }
}

// Implement cuda_malloc
template<typename T>
std::unique_ptr<T, cuda_deleter> cuda_malloc(size_t size) {
    T* ptr = nullptr;
    CHECK_CUDA(cudaMalloc(&ptr, size * sizeof(T)));
    return std::unique_ptr<T, cuda_deleter>(ptr);
}

// Implement NeuralNetwork methods
NeuralNetwork::NeuralNetwork(float initial_lr) : 
    d_input(cuda_malloc<float>(BATCH_SIZE * INPUT_SIZE)),
    d_hidden(cuda_malloc<float>(BATCH_SIZE * HIDDEN_SIZE)),
    d_output(cuda_malloc<float>(BATCH_SIZE * OUTPUT_SIZE)),
    d_W1(cuda_malloc<float>(INPUT_SIZE * HIDDEN_SIZE)),
    d_W2(cuda_malloc<float>(HIDDEN_SIZE * OUTPUT_SIZE)),
    d_b1(cuda_malloc<float>(HIDDEN_SIZE)),
    d_b2(cuda_malloc<float>(OUTPUT_SIZE)),
    d_dW1(cuda_malloc<float>(INPUT_SIZE * HIDDEN_SIZE)),
    d_dW2(cuda_malloc<float>(HIDDEN_SIZE * OUTPUT_SIZE)),
    d_db1(cuda_malloc<float>(HIDDEN_SIZE)),
    d_db2(cuda_malloc<float>(OUTPUT_SIZE)),
    d_labels(cuda_malloc<float>(BATCH_SIZE * OUTPUT_SIZE)),
    d_loss(cuda_malloc<float>(1)),
    d_d_logits(cuda_malloc<float>(BATCH_SIZE * OUTPUT_SIZE)),
    d_hidden_derivative(cuda_malloc<float>(BATCH_SIZE * HIDDEN_SIZE)),
    learning_rate(initial_lr)
{
    CHECK_CUBLAS(cublasCreate(&cublas_handle));
    CHECK_CURAND(curandCreateGenerator(&curand_gen, CURAND_RNG_PSEUDO_DEFAULT));
    CHECK_CURAND(curandSetPseudoRandomGeneratorSeed(curand_gen, 1234ULL));

    CHECK_CUDA(cudaStreamCreate(&train_stream));
    CHECK_CUDA(cudaStreamCreate(&val_stream));

    initialize_weights();
}

NeuralNetwork::~NeuralNetwork() {
    CHECK_CUBLAS(cublasDestroy(cublas_handle));
    CHECK_CURAND(curandDestroyGenerator(curand_gen));
    CHECK_CUDA(cudaStreamDestroy(train_stream));
    CHECK_CUDA(cudaStreamDestroy(val_stream));
}

void NeuralNetwork::initialize_weights() {
    float stddev_W1 = sqrtf(2.0f / (INPUT_SIZE + HIDDEN_SIZE));
    CHECK_CURAND(curandGenerateNormal(curand_gen, d_W1.get(), INPUT_SIZE * HIDDEN_SIZE, 0.0f, stddev_W1));

    float stddev_W2 = sqrtf(2.0f / (HIDDEN_SIZE + OUTPUT_SIZE));
    CHECK_CURAND(curandGenerateNormal(curand_gen, d_W2.get(), HIDDEN_SIZE * OUTPUT_SIZE, 0.0f, stddev_W2));

    CHECK_CUDA(cudaMemsetAsync(d_b1.get(), 0, HIDDEN_SIZE * sizeof(float), train_stream));
    CHECK_CUDA(cudaMemsetAsync(d_b2.get(), 0, OUTPUT_SIZE * sizeof(float), train_stream));
}

void NeuralNetwork::forward(const float* input, bool is_validation) {
    cudaStream_t current_stream = is_validation ? val_stream : train_stream;

    CHECK_CUDA(cudaMemcpyAsync(d_input.get(), input, BATCH_SIZE * INPUT_SIZE * sizeof(float), cudaMemcpyHostToDevice, current_stream));

    float alpha = 1.0f, beta = 0.0f;
    CHECK_CUBLAS(cublasSetStream(cublas_handle, current_stream));
    CHECK_CUBLAS(cublasSgemm(
        cublas_handle,
        CUBLAS_OP_T,
        CUBLAS_OP_N,
        HIDDEN_SIZE,
        BATCH_SIZE,
        INPUT_SIZE,
        &alpha,
        d_W1.get(),
        HIDDEN_SIZE,
        d_input.get(),
        INPUT_SIZE,
        &beta,
        d_hidden.get(),
        HIDDEN_SIZE
    ));

    int total_hidden = BATCH_SIZE * HIDDEN_SIZE;
    int block_size = 256;
    int grid_size = (total_hidden + block_size - 1) / block_size;
    fused_bias_relu<<<grid_size, block_size, 0, current_stream>>>(
        d_hidden.get(),
        d_b1.get(),
        BATCH_SIZE,
        HIDDEN_SIZE
    );
    CHECK_CUDA(cudaGetLastError());

    CHECK_CUBLAS(cublasSgemm(
        cublas_handle,
        CUBLAS_OP_T,
        CUBLAS_OP_N,
        OUTPUT_SIZE,
        BATCH_SIZE,
        HIDDEN_SIZE,
        &alpha,
        d_W2.get(),
        OUTPUT_SIZE,
        d_hidden.get(),
        HIDDEN_SIZE,
        &beta,
        d_output.get(),
        OUTPUT_SIZE
    ));

    int total_output = BATCH_SIZE * OUTPUT_SIZE;
    grid_size = (total_output + block_size - 1) / block_size;
    fused_bias_relu<<<grid_size, block_size, 0, current_stream>>>(
        d_output.get(),
        d_b2.get(),
        BATCH_SIZE,
        OUTPUT_SIZE
    );
    CHECK_CUDA(cudaGetLastError());
}

float NeuralNetwork::backward(const float* labels, bool is_validation) {
    if (is_validation) return 0.0f;

    CHECK_CUDA(cudaMemcpyAsync(d_labels.get(), labels, BATCH_SIZE * OUTPUT_SIZE * sizeof(float), cudaMemcpyHostToDevice, train_stream));
    CHECK_CUDA(cudaMemsetAsync(d_loss.get(), 0, sizeof(float), train_stream));

    int grid_size = BATCH_SIZE;
    int block_size = 1;
    size_t shared_mem_size = 0;
    softmax_cross_entropy<<<grid_size, block_size, shared_mem_size, train_stream>>>(
        d_output.get(),
        d_labels.get(),
        d_loss.get(),
        d_d_logits.get(),
        BATCH_SIZE,
        OUTPUT_SIZE
    );
    CHECK_CUDA(cudaGetLastError());

    float loss = 0.0f;
    CHECK_CUDA(cudaMemcpyAsync(&loss, d_loss.get(), sizeof(float), cudaMemcpyDeviceToHost, train_stream));
    CHECK_CUDA(cudaStreamSynchronize(train_stream));
    loss /= BATCH_SIZE;

    float alpha = 1.0f / BATCH_SIZE;
    float beta = 0.0f;
    CHECK_CUBLAS(cublasSetStream(cublas_handle, train_stream));
    CHECK_CUBLAS(cublasSgemm(
        cublas_handle,
        CUBLAS_OP_N,
        CUBLAS_OP_T,
        HIDDEN_SIZE,
        OUTPUT_SIZE,
        BATCH_SIZE,
        &alpha,
        d_hidden.get(),
        HIDDEN_SIZE,
        d_d_logits.get(),
        OUTPUT_SIZE,
        &beta,
        d_dW2.get(),
        HIDDEN_SIZE
    ));

    CHECK_CUDA(cudaMemsetAsync(d_db2.get(), 0, OUTPUT_SIZE * sizeof(float), train_stream));
    dim3 db2_grid((OUTPUT_SIZE + 255) / 256);
    dim3 db2_block(256);
    sum_gradients<<<OUTPUT_SIZE, 256, 0, train_stream>>>(d_d_logits.get(), d_db2.get(), BATCH_SIZE, OUTPUT_SIZE);
    CHECK_CUDA(cudaGetLastError());

    CHECK_CUBLAS(cublasSgemm(
        cublas_handle,
        CUBLAS_OP_N,
        CUBLAS_OP_T,
        HIDDEN_SIZE,
        BATCH_SIZE,
        OUTPUT_SIZE,
        &alpha,
        d_W2.get(),
        OUTPUT_SIZE,
        d_d_logits.get(),
        OUTPUT_SIZE,
        &beta,
        d_hidden_derivative.get(),
        HIDDEN_SIZE
    ));

    int total_hidden = BATCH_SIZE * HIDDEN_SIZE;
    int relu_grid_size = (total_hidden + 256 - 1) / 256;
    gradient_clipping<<<relu_grid_size, 256, 0, train_stream>>>(
        d_hidden_derivative.get(),
        total_hidden,
        CLIP_THRESHOLD
    );
    CHECK_CUDA(cudaGetLastError());

    CHECK_CUBLAS(cublasSgemm(
        cublas_handle,
        CUBLAS_OP_T,
        CUBLAS_OP_N,
        HIDDEN_SIZE,
        INPUT_SIZE,
        BATCH_SIZE,
        &alpha,
        d_hidden_derivative.get(),
        HIDDEN_SIZE,
        d_input.get(),
        INPUT_SIZE,
        &beta,
        d_dW1.get(),
        HIDDEN_SIZE
    ));

    CHECK_CUDA(cudaMemsetAsync(d_db1.get(), 0, HIDDEN_SIZE * sizeof(float), train_stream));
    dim3 db1_grid((HIDDEN_SIZE + 255) / 256);
    dim3 db1_block(256);
    sum_gradients<<<HIDDEN_SIZE, 256, 0, train_stream>>>(d_hidden_derivative.get(), d_db1.get(), BATCH_SIZE, HIDDEN_SIZE);
    CHECK_CUDA(cudaGetLastError());

    int grad_size_W1 = INPUT_SIZE * HIDDEN_SIZE;
    int grad_size_W2 = HIDDEN_SIZE * OUTPUT_SIZE;
    int clip_block = 256;
    int clip_grid = (grad_size_W1 + clip_block - 1) / clip_block;
    gradient_clipping<<<clip_grid, clip_block, 0, train_stream>>>(
        d_dW1.get(),
        grad_size_W1,
        CLIP_THRESHOLD
    );
    CHECK_CUDA(cudaGetLastError());

    clip_grid = (grad_size_W2 + clip_block - 1) / clip_block;
    gradient_clipping<<<clip_grid, clip_block, 0, train_stream>>>(
        d_dW2.get(),
        grad_size_W2,
        CLIP_THRESHOLD
    );
    CHECK_CUDA(cudaGetLastError());

    CHECK_CUDA(cudaStreamSynchronize(train_stream));

    return loss;
}

void NeuralNetwork::update_weights() {
    float neg_lr = -learning_rate;

    CHECK_CUBLAS(cublasSetStream(cublas_handle, train_stream));
    CHECK_CUBLAS(cublasSaxpy(
        cublas_handle,
        INPUT_SIZE * HIDDEN_SIZE,
        &neg_lr,
        d_dW1.get(),
        1,
        d_W1.get(),
        1
    ));

    CHECK_CUBLAS(cublasSaxpy(
        cublas_handle,
        HIDDEN_SIZE * OUTPUT_SIZE,
        &neg_lr,
        d_dW2.get(),
        1,
        d_W2.get(),
        1
    ));

    CHECK_CUBLAS(cublasSaxpy(
        cublas_handle,
        HIDDEN_SIZE,
        &neg_lr,
        d_db1.get(),
        1,
        d_b1.get(),
        1
    ));

    CHECK_CUBLAS(cublasSaxpy(
        cublas_handle,
        OUTPUT_SIZE,
        &neg_lr,
        d_db2.get(),
        1,
        d_b2.get(),
        1
    ));
}

void NeuralNetwork::save_model(const std::string& filename) {
    std::ofstream ofs(filename, std::ios::binary);
    if (!ofs) {
        throw std::runtime_error("Failed to open file for saving model.");
    }

    std::vector<float> h_W1(INPUT_SIZE * HIDDEN_SIZE);
    std::vector<float> h_W2(HIDDEN_SIZE * OUTPUT_SIZE);
    std::vector<float> h_b1(HIDDEN_SIZE);
    std::vector<float> h_b2(OUTPUT_SIZE);

    CHECK_CUDA(cudaMemcpy(h_W1.data(), d_W1.get(), INPUT_SIZE * HIDDEN_SIZE * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_W2.data(), d_W2.get(), HIDDEN_SIZE * OUTPUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_b1.data(), d_b1.get(), HIDDEN_SIZE * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_b2.data(), d_b2.get(), OUTPUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost));

    ofs.write(reinterpret_cast<char*>(h_W1.data()), INPUT_SIZE * HIDDEN_SIZE * sizeof(float));
    ofs.write(reinterpret_cast<char*>(h_W2.data()), HIDDEN_SIZE * OUTPUT_SIZE * sizeof(float));
    ofs.write(reinterpret_cast<char*>(h_b1.data()), HIDDEN_SIZE * sizeof(float));
    ofs.write(reinterpret_cast<char*>(h_b2.data()), OUTPUT_SIZE * sizeof(float));

    ofs.close();
    std::cout << "Model saved to " << filename << std::endl;
}

void NeuralNetwork::load_model(const std::string& filename) {
    std::ifstream ifs(filename, std::ios::binary);
    if (!ifs) {
        throw std::runtime_error("Failed to open file for loading model.");
    }

    std::vector<float> h_W1(INPUT_SIZE * HIDDEN_SIZE);
    std::vector<float> h_W2(HIDDEN_SIZE * OUTPUT_SIZE);
    std::vector<float> h_b1(HIDDEN_SIZE);
    std::vector<float> h_b2(OUTPUT_SIZE);

    ifs.read(reinterpret_cast<char*>(h_W1.data()), INPUT_SIZE * HIDDEN_SIZE * sizeof(float));
    ifs.read(reinterpret_cast<char*>(h_W2.data()), HIDDEN_SIZE * OUTPUT_SIZE * sizeof(float));
    ifs.read(reinterpret_cast<char*>(h_b1.data()), HIDDEN_SIZE * sizeof(float));
    ifs.read(reinterpret_cast<char*>(h_b2.data()), OUTPUT_SIZE * sizeof(float));

    ifs.close();

    CHECK_CUDA(cudaMemcpy(d_W1.get(), h_W1.data(), INPUT_SIZE * HIDDEN_SIZE * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_W2.get(), h_W2.data(), HIDDEN_SIZE * OUTPUT_SIZE * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_b1.get(), h_b1.data(), HIDDEN_SIZE * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_b2.get(), h_b2.data(), OUTPUT_SIZE * sizeof(float), cudaMemcpyHostToDevice));

    std::cout << "Model loaded from " << filename << std::endl;
}

void NeuralNetwork::adjust_learning_rate(int epoch, float decay_factor) {
    learning_rate *= decay_factor;
    std::cout << "Learning rate adjusted to " << learning_rate << std::endl;
}

// Helper functions
void generate_one_hot_labels(std::vector<float>& labels, int batch_size, int output_size) {
    std::fill(labels.begin(), labels.end(), 0.0f);
    for (int i = 0; i < batch_size; ++i) {
        int label = rand() % output_size;
        labels[i * output_size + label] = 1.0f;
    }
}

void load_batch_data(std::vector<float>& input, std::vector<float>& labels, int batch_size, int input_size, int output_size) {
    for (int i = 0; i < batch_size * input_size; ++i) {
        input[i] = static_cast<float>(rand()) / RAND_MAX;
    }

    generate_one_hot_labels(labels, batch_size, output_size);
}

float validate(NeuralNetwork& nn, int num_validation_batches) {
    float total_loss = 0.0f;
    std::vector<float> input(BATCH_SIZE * INPUT_SIZE);
    std::vector<float> labels(BATCH_SIZE * OUTPUT_SIZE);

    for (int batch = 0; batch < num_validation_batches; ++batch) {
        load_batch_data(input, labels, BATCH_SIZE, INPUT_SIZE, OUTPUT_SIZE);

        nn.forward(input.data(), true);

        CHECK_CUDA(cudaMemset(nn.get_d_loss(), 0, sizeof(float)));

        int grid_size = BATCH_SIZE;
        int block_size = 1;
        size_t shared_mem_size = 0;
        softmax_cross_entropy<<<grid_size, block_size, shared_mem_size, nn.val_stream>>>(
            nn.get_d_output(),
            nn.get_d_labels(),
            nn.get_d_loss(),
            nn.get_d_d_logits(),
            BATCH_SIZE,
            OUTPUT_SIZE
        );
        CHECK_CUDA(cudaGetLastError());

        float loss = 0.0f;
        CHECK_CUDA(cudaMemcpyAsync(&loss, nn.get_d_loss(), sizeof(float), cudaMemcpyDeviceToHost, nn.val_stream));
        CHECK_CUDA(cudaStreamSynchronize(nn.val_stream));
        loss /= BATCH_SIZE;

        total_loss += loss;
    }

    return total_loss / num_validation_batches;
}