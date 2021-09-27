#include "class_keypoints_.h"
#include "NvInfer.h"
#include "cuda_runtime_api.h"
#include "cuda_utils.h"
#include "logging.h"

using namespace nvinfer1;

static Logger gLogger;

#define USE_FP16
#define DEVICE (0)
static const int INPUT_H = 256;
static const int INPUT_W = 192;
static const int BATCH_SIZE = 4;
static const int HEATMAP_C = 17;
static const int HEATMAP_H = 64;
static const int HEATMAP_W = 48;
static const int OUTPUT_SIZE = HEATMAP_C * HEATMAP_H * HEATMAP_W; // CHW
static const char* INPUT_BLOB_NAME = "input";
static const char* OUTPUT_BLOB_NAME = "output";


static std::map<std::string, Weights> loadWeights(const std::string file)
{
    std::cout << "Loading weights: " << file << std::endl;
    std::map<std::string, Weights> weightMap;

    // Open weights file
    std::ifstream input(file);
    assert(input.is_open() && "Unable to load weight file.");

    // Read number of weight blobs
    int32_t count = 0;
    input >> count;
    assert(count > 0 && "Invalid weight map file.");

    while (count--)
    {
        Weights wt{DataType::kFLOAT, nullptr, 0};
        uint32_t size;

        // Read name and type of blob
        std::string name;
        input >> name >> std::dec >> size;
        wt.type = DataType::kFLOAT;

        // Load blob
        uint32_t* val = reinterpret_cast<uint32_t*>(malloc(sizeof(val) * size));
        for (uint32_t x = 0, y = size; x < y; ++x)
        {
            input >> std::hex >> val[x];
        }
        wt.values = val;
        
        wt.count = size;
        weightMap[name] = wt;
    }

    return weightMap;
}

static IScaleLayer* addBatchNorm2d(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor& input, std::string lname, float eps) 
{
    float *gamma = (float*)weightMap[lname + ".weight"].values;
    float *beta = (float*)weightMap[lname + ".bias"].values;
    float *mean = (float*)weightMap[lname + ".running_mean"].values;
    float *var = (float*)weightMap[lname + ".running_var"].values;
    int len = weightMap[lname + ".running_var"].count;
    // std::cout << "len " << len << std::endl;

    float *scval = reinterpret_cast<float*>(malloc(sizeof(float) * len));
    for (int i = 0; i < len; i++) {
        scval[i] = gamma[i] / sqrt(var[i] + eps);
    }
    Weights scale{DataType::kFLOAT, scval, len};
    
    float *shval = reinterpret_cast<float*>(malloc(sizeof(float) * len));
    for (int i = 0; i < len; i++) {
        shval[i] = beta[i] - mean[i] * gamma[i] / sqrt(var[i] + eps);
    }
    Weights shift{DataType::kFLOAT, shval, len};

    float *pval = reinterpret_cast<float*>(malloc(sizeof(float) * len));
    for (int i = 0; i < len; i++) {
        pval[i] = 1.0;
    }
    Weights power{DataType::kFLOAT, pval, len};

    weightMap[lname + ".scale"] = scale;
    weightMap[lname + ".shift"] = shift;
    weightMap[lname + ".power"] = power;
    IScaleLayer* scale_1 = network->addScale(input, ScaleMode::kCHANNEL, shift, scale, power);
    assert(scale_1);
    return scale_1;
}

static IActivationLayer* bottleneck(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor& input, int inch, int outch, int stride, std::string lname) 
{
    Weights emptywts{DataType::kFLOAT, nullptr, 0};

    IConvolutionLayer* conv1 = network->addConvolutionNd(input, outch, DimsHW{1, 1}, weightMap[lname + "conv1.weight"], emptywts);
    assert(conv1);

    IScaleLayer* bn1 = addBatchNorm2d(network, weightMap, *conv1->getOutput(0), lname + "bn1", 1e-5);

    IActivationLayer* relu1 = network->addActivation(*bn1->getOutput(0), ActivationType::kRELU);
    assert(relu1);

    IConvolutionLayer* conv2 = network->addConvolutionNd(*relu1->getOutput(0), outch, DimsHW{3, 3}, weightMap[lname + "conv2.weight"], emptywts);
    assert(conv2);
    conv2->setStrideNd(DimsHW{stride, stride});
    conv2->setPaddingNd(DimsHW{1, 1});

    IScaleLayer* bn2 = addBatchNorm2d(network, weightMap, *conv2->getOutput(0), lname + "bn2", 1e-5);

    IActivationLayer* relu2 = network->addActivation(*bn2->getOutput(0), ActivationType::kRELU);
    assert(relu2);

    IConvolutionLayer* conv3 = network->addConvolutionNd(*relu2->getOutput(0), outch * 4, DimsHW{1, 1}, weightMap[lname + "conv3.weight"], emptywts);
    assert(conv3);

    IScaleLayer* bn3 = addBatchNorm2d(network, weightMap, *conv3->getOutput(0), lname + "bn3", 1e-5);

    IElementWiseLayer* ew1;
    if (stride != 1 || inch != outch * 4) {
        IConvolutionLayer* conv4 = network->addConvolutionNd(input, outch * 4, DimsHW{1, 1}, weightMap[lname + "downsample.0.weight"], emptywts);
        assert(conv4);
        conv4->setStrideNd(DimsHW{stride, stride});

        IScaleLayer* bn4 = addBatchNorm2d(network, weightMap, *conv4->getOutput(0), lname + "downsample.1", 1e-5);
        ew1 = network->addElementWise(*bn4->getOutput(0), *bn3->getOutput(0), ElementWiseOperation::kSUM);
    } else {
        ew1 = network->addElementWise(input, *bn3->getOutput(0), ElementWiseOperation::kSUM);
    }
    IActivationLayer* relu3 = network->addActivation(*ew1->getOutput(0), ActivationType::kRELU);
    assert(relu3);
    return relu3;
}

static ICudaEngine* createEngine(unsigned int maxBatchSize, 
                          IBuilder* builder, 
                          IBuilderConfig* config, 
                          DataType dt, 
                          const std::string &wts_path_)
{
    INetworkDefinition* network = builder->createNetworkV2(0U);

    // Create input tensor of shape { 3, INPUT_H, INPUT_W } with name INPUT_BLOB_NAME
    ITensor* data = network->addInput(INPUT_BLOB_NAME, dt, Dims3{3, INPUT_H, INPUT_W});
    assert(data);

    std::map<std::string, Weights> weightMap = loadWeights(wts_path_);
    Weights emptywts{DataType::kFLOAT, nullptr, 0};

    IConvolutionLayer* conv1 = network->addConvolutionNd(*data, 64, DimsHW{7, 7}, weightMap["preact.conv1.weight"], emptywts);
    assert(conv1);
    conv1->setStrideNd(DimsHW{2, 2});
    conv1->setPaddingNd(DimsHW{3, 3});

    IScaleLayer* bn1 = addBatchNorm2d(network, weightMap, *conv1->getOutput(0), "preact.bn1", 1e-5);

    // Add activation layer using the ReLU algorithm.
    IActivationLayer* relu1 = network->addActivation(*bn1->getOutput(0), ActivationType::kRELU);
    assert(relu1);

    // Add max pooling layer with stride of 2x2 and kernel size of 2x2.
    IPoolingLayer* pool1 = network->addPoolingNd(*relu1->getOutput(0), PoolingType::kMAX, DimsHW{3, 3});
    assert(pool1);
    pool1->setStrideNd(DimsHW{2, 2});
    pool1->setPaddingNd(DimsHW{1, 1});

    IActivationLayer* x = bottleneck(network, weightMap, *pool1->getOutput(0), 64, 64, 1, "preact.layer1.0.");
    x = bottleneck(network, weightMap, *x->getOutput(0), 256, 64, 1, "preact.layer1.1.");
    x = bottleneck(network, weightMap, *x->getOutput(0), 256, 64, 1, "preact.layer1.2.");

    x = bottleneck(network, weightMap, *x->getOutput(0), 256, 128, 2, "preact.layer2.0.");
    x = bottleneck(network, weightMap, *x->getOutput(0), 512, 128, 1, "preact.layer2.1.");
    x = bottleneck(network, weightMap, *x->getOutput(0), 512, 128, 1, "preact.layer2.2.");
    x = bottleneck(network, weightMap, *x->getOutput(0), 512, 128, 1, "preact.layer2.3.");

    x = bottleneck(network, weightMap, *x->getOutput(0), 512, 256, 2, "preact.layer3.0.");
    x = bottleneck(network, weightMap, *x->getOutput(0), 1024, 256, 1, "preact.layer3.1.");
    x = bottleneck(network, weightMap, *x->getOutput(0), 1024, 256, 1, "preact.layer3.2.");
    x = bottleneck(network, weightMap, *x->getOutput(0), 1024, 256, 1, "preact.layer3.3.");
    x = bottleneck(network, weightMap, *x->getOutput(0), 1024, 256, 1, "preact.layer3.4.");
    x = bottleneck(network, weightMap, *x->getOutput(0), 1024, 256, 1, "preact.layer3.5.");

    x = bottleneck(network, weightMap, *x->getOutput(0), 1024, 512, 2, "preact.layer4.0.");
    x = bottleneck(network, weightMap, *x->getOutput(0), 2048, 512, 1, "preact.layer4.1.");
    x = bottleneck(network, weightMap, *x->getOutput(0), 2048, 512, 1, "preact.layer4.2.");

    // for (int i = 0; i < 3 ;i++)
    // {
    //     std::cout << x->getOutput(0)->getDimensions().d[i] << " ";
    // }
    // std::cout << std::endl;

    // IPoolingLayer* pool2 = network->addPoolingNd(*x->getOutput(0), PoolingType::kAVERAGE, DimsHW{7, 7});
    // assert(pool2);
    // pool2->setStrideNd(DimsHW{1, 1});

    std::cout << "resnet50 create" << std::endl;

    //
    Weights emptywts_deconv1{DataType::kFLOAT, nullptr, 0};
    IDeconvolutionLayer* deconv1 = network->addDeconvolutionNd(*x->getOutput(0), 256, DimsHW{4, 4}, weightMap["deconv_layers.0.weight"], emptywts_deconv1);
    assert(deconv1);
    deconv1->setStrideNd(DimsHW{2, 2});
    // deconv1->setNbGroups(256);
    deconv1->setPaddingNd(DimsHW{1, 1});
    IScaleLayer* bn_deconv1 = addBatchNorm2d(network, weightMap, *deconv1->getOutput(0), "deconv_layers.1", 1e-5);
    assert(bn_deconv1);
    IActivationLayer* relu_deconv1 = network->addActivation(*bn_deconv1->getOutput(0), ActivationType::kRELU);
    assert(relu_deconv1);

    
    //
    Weights emptywts_deconv2{DataType::kFLOAT, nullptr, 0};
    IDeconvolutionLayer* deconv2 = network->addDeconvolutionNd(*relu_deconv1->getOutput(0), 256, DimsHW{4, 4}, weightMap["deconv_layers.3.weight"], emptywts_deconv2);
    assert(deconv2);
    deconv2->setStrideNd(DimsHW{2, 2});
    // deconv2->setNbGroups(256);
    deconv2->setPaddingNd(DimsHW{1, 1});
    IScaleLayer* bn_deconv2 = addBatchNorm2d(network, weightMap, *deconv2->getOutput(0), "deconv_layers.4", 1e-5);
    assert(bn_deconv2);
    IActivationLayer* relu_deconv2 = network->addActivation(*bn_deconv2->getOutput(0), ActivationType::kRELU);
    assert(relu_deconv2);

    //
    Weights emptywts_deconv3{DataType::kFLOAT, nullptr, 0};
    IDeconvolutionLayer* deconv3 = network->addDeconvolutionNd(*relu_deconv2->getOutput(0), 256, DimsHW{4, 4}, weightMap["deconv_layers.6.weight"], emptywts_deconv3);
    assert(deconv3);
    deconv3->setStrideNd(DimsHW{2, 2});
    // deconv3->setNbGroups(256);
    deconv3->setPaddingNd(DimsHW{1, 1});
    IScaleLayer* bn_deconv3 = addBatchNorm2d(network, weightMap, *deconv3->getOutput(0), "deconv_layers.7", 1e-5);
    assert(bn_deconv3);
    IActivationLayer* relu_deconv3 = network->addActivation(*bn_deconv3->getOutput(0), ActivationType::kRELU);
    assert(relu_deconv3);

    //
    IConvolutionLayer* final_conv = network->addConvolutionNd(*relu_deconv3->getOutput(0), 17, DimsHW{1, 1}, weightMap["final_layer.weight"], weightMap["final_layer.bias"]);
    final_conv->setStrideNd(DimsHW{1, 1});

    //
    final_conv->getOutput(0)->setName(OUTPUT_BLOB_NAME);
    std::cout << "set name out" << std::endl;
    network->markOutput(*final_conv->getOutput(0));

    // Build engine
    builder->setMaxBatchSize(maxBatchSize);
    config->setMaxWorkspaceSize(16 * (1 << 20));
#if defined(USE_FP16)
    config->setFlag(BuilderFlag::kFP16);
#endif
    ICudaEngine* engine = builder->buildEngineWithConfig(*network, *config);
    std::cout << "build out" << std::endl;

    // Don't need the network any more
    network->destroy();

    // Release host memory
    for (auto& mem : weightMap)
    {
        free((void*) (mem.second.values));
    }

    return engine;
}

static void APIToModel(unsigned int maxBatchSize, IHostMemory** modelStream, const std::string &wts_path_)
{
    // Create builder
    IBuilder* builder = createInferBuilder(gLogger);
    IBuilderConfig* config = builder->createBuilderConfig();

    // Create model to populate the network, then set the outputs and create an engine
    ICudaEngine* engine = createEngine(maxBatchSize, builder, config, DataType::kFLOAT, wts_path_);
    assert(engine != nullptr);

    // Serialize the engine
    (*modelStream) = engine->serialize();

    // Close everything down
    engine->destroy();
    builder->destroy();
    config->destroy();
}

static void doInference(IExecutionContext& context, cudaStream_t& stream, void **buffers, float* input, float* output, int batchSize) {
    // DMA input batch data to device, infer on the batch asynchronously, and DMA output back to host
    CUDA_CHECK(cudaMemcpyAsync(buffers[0], input, batchSize * 3 * INPUT_H * INPUT_W * sizeof(float), cudaMemcpyHostToDevice, stream));
    context.enqueue(batchSize, buffers, stream, nullptr);
    CUDA_CHECK(cudaMemcpyAsync(output, buffers[1], batchSize * OUTPUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost, stream));
    cudaStreamSynchronize(stream);
}


static void preprocess_img(const cv::Mat &img_, cv::Mat &img_preprocess_)
{
    cv::Mat img_preprocess = cv::Mat(INPUT_H, INPUT_W, CV_8UC3);
    cv::resize(img_, img_preprocess, img_preprocess.size());
    img_preprocess_ = img_preprocess.clone();
}

static void post_process(float* output_, int b, int orig_cols_, int orig_rows_, std::vector<cv::Point2f> &coords_, std::vector<float> &scores_)
{
    float *features = output_ + b * OUTPUT_SIZE;
    for (int i = 0; i < HEATMAP_C; i++)
    {
        float *feature = features + i * HEATMAP_H * HEATMAP_W;
        cv::Mat img = cv::Mat(HEATMAP_H, HEATMAP_W, CV_32FC1, feature);

        double min_value, max_value;
        cv::Point2i min_loc, max_loc;
        cv::minMaxLoc(img, &min_value, &max_value, &min_loc, &max_loc);
        
        cv::Point2f coord(max_loc.x * 1.0 * orig_cols_ / HEATMAP_W, max_loc.y * 1.0 * orig_rows_ / HEATMAP_H);
        coords_.push_back(coord);
        scores_.push_back(max_value);
    }
}

_Keypoints::_Keypoints() {}
_Keypoints::~_Keypoints() 
{
    if (_buffers[0])
    {
        CUDA_CHECK(cudaFree(_buffers[0]));
    }
    if (_buffers[1])
    {
        CUDA_CHECK(cudaFree(_buffers[1]));
    }
    
    // Destroy the engine
    if (_context)
    {
        _context->destroy();
    }
    
    if (_engine)
    {
        _engine->destroy();
    }
    
    if (_runtime)
    {
        _runtime->destroy();
    }
}

bool _Keypoints::serialize(const std::string &wts_path_, const std::string &engine_path_)
{
    IHostMemory* modelStream{nullptr};
    APIToModel(BATCH_SIZE, &modelStream, wts_path_);
    assert(modelStream != nullptr);

    std::ofstream p(engine_path_, std::ios::binary);
    if (!p)
    {
        std::cerr << "could not open plan output file" << std::endl;
        return false;
    }
    p.write(reinterpret_cast<const char*>(modelStream->data()), modelStream->size());
    modelStream->destroy();
    return true;
}
 

bool _Keypoints::init(const std::string &engine_path_)
{
    cudaSetDevice(DEVICE);

    std::ifstream file(engine_path_, std::ios::binary);
    if (!file.good()) {
        std::cerr << "read " << engine_path_ << " error!" << std::endl;
        return false;
    }

    char *trtModelStream = nullptr;
    size_t size = 0;
    file.seekg(0, file.end);
    size = file.tellg();
    file.seekg(0, file.beg);
    trtModelStream = new char[size];
    assert(trtModelStream);
    file.read(trtModelStream, size);
    file.close();

    _runtime = createInferRuntime(gLogger);
    assert(_runtime != nullptr);
    _engine = _runtime->deserializeCudaEngine(trtModelStream, size);
    assert(_engine != nullptr);
    _context = _engine->createExecutionContext();
    assert(_context != nullptr);
    delete[] trtModelStream;
    assert(_engine->getNbBindings() == 2);
    
    // In order to bind the buffers, we need to know the names of the input and output tensors.
    // Note that indices are guaranteed to be less than IEngine::getNbBindings()
    const int inputIndex = _engine->getBindingIndex(INPUT_BLOB_NAME);
    const int outputIndex = _engine->getBindingIndex(OUTPUT_BLOB_NAME);
    assert(inputIndex == 0);
    assert(outputIndex == 1);
    
    // Create GPU buffers on device
    CUDA_CHECK(cudaMalloc(&_buffers[inputIndex], BATCH_SIZE * 3 * INPUT_H * INPUT_W * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&_buffers[outputIndex], BATCH_SIZE * OUTPUT_SIZE * sizeof(float)));
    
    // Create stream
    CUDA_CHECK(cudaStreamCreate(&_stream));

    return true;
}

bool _Keypoints::run(const std::vector<cv::Mat> &imgs_, 
               std::vector<std::vector<cv::Point2f> > &vec_coords_,
               std::vector<std::vector<float> > &vec_scores_)
{
    static float data[BATCH_SIZE * 3 * INPUT_H * INPUT_W];

    static float output[BATCH_SIZE * OUTPUT_SIZE];


    int fcount = 0;
    for (int f = 0; f < (int)imgs_.size(); f++) 
    {
        fcount++;
        if (fcount < BATCH_SIZE && f + 1 != (int)imgs_.size()) 
        {
            continue;
        }

        // auto start1 = std::chrono::system_clock::now();

        for (int b = 0; b < fcount; b++) 
        {
            cv::Mat img = imgs_[f - fcount + 1 + b];
            
            // 
            cv::Mat img_preprocess;
            preprocess_img(img, img_preprocess);
                
            int i = 0;
            for (int row = 0; row < INPUT_H; ++row) 
            {
                uchar* uc_pixel = img_preprocess.data + row * img_preprocess.step;
                for (int col = 0; col < INPUT_W; ++col) 
                {
                    // data[b * 3 * INPUT_H * INPUT_W + i] = ((float)uc_pixel[2] / 255.0 - 0.480) / 0.229;
                    // data[b * 3 * INPUT_H * INPUT_W + i + INPUT_H * INPUT_W] = ((float)uc_pixel[1] / 255.0 - 0.457) / 0.224;
                    // data[b * 3 * INPUT_H * INPUT_W + i + 2 * INPUT_H * INPUT_W] = ((float)uc_pixel[0] / 255.0 - 0.406) / 0.225;
                    data[b * 3 * INPUT_H * INPUT_W + i] = ((float)uc_pixel[2] / 255.0 - 0.406);
                    data[b * 3 * INPUT_H * INPUT_W + i + INPUT_H * INPUT_W] = ((float)uc_pixel[1] / 255.0 - 0.457);
                    data[b * 3 * INPUT_H * INPUT_W + i + 2 * INPUT_H * INPUT_W] = ((float)uc_pixel[0] / 255.0 - 0.480);
                    uc_pixel += 3;
                    ++i;
                }
            }
        }

        // auto end1 = std::chrono::system_clock::now();
        // std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(end1 - start1).count() << " ms" << std::endl;
        

        // Run inference
        // auto start = std::chrono::system_clock::now();
        
        doInference(*_context, _stream, _buffers, data, output, BATCH_SIZE);
        
        // auto end = std::chrono::system_clock::now();
        // std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << " ---ms" << std::endl;
        
        for (int b = 0; b < fcount; b++) 
        {
            cv::Mat img = imgs_[f - fcount + 1 + b];
            //
            std::vector<cv::Point2f> coords;
            std::vector<float> scores;

            post_process(output, b, img.cols, img.rows, coords, scores);
            
            vec_coords_.push_back(coords);
            vec_scores_.push_back(scores);
        }

        //
        fcount = 0;
    }

    return true;
}

