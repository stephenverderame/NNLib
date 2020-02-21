struct net_info{
    unsigned int layerNum, layers[10], dataNum, maxLayer;
    double learningRate;
};
__kernel void train(__constant double * weights, __constant double * biases, __constant net_info * info, __constant double * inputs, __constant double * reals, __global double * trainedParams){
    unsigned int inputSize = info->layers[0];
    unsigned int outputSize = info->layers[info->layerNum - 1];
    int i = get_group_id(0);
    int j = get_global_id(0);

}