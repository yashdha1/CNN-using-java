package Nets;

import Layers.*;
import java.util.List;
import java.util.ArrayList;

public class NetBuilder {
    private NeuralNetwork net;
    private int inRow; 
    private int inCol;
    List<Layer> layers; 
    private double SF; 

    public NetBuilder(int inRow, int inCol, double SF) {
        this.inRow = inRow;
        this.inCol = inCol;
        this.SF = SF;
        layers = new ArrayList<>();
    }

    public void addConvulutionLayer(int numFilter, int filterSize, int stepSize, double LR, long SEED) {
        if(layers.isEmpty()){
            layers.add(new ConvolutionLayer(filterSize, stepSize, 1, inRow, inCol,  SEED, numFilter, LR)) ; 
        }else{
            Layer prevLayer = layers.get(layers.size()-1);
            layers.add(new ConvolutionLayer(filterSize, stepSize, prevLayer.getOutputLen(), prevLayer.getOutputRow(), prevLayer.getOutputCol(), SEED, numFilter, LR)) ;
        }
    }
    public void addMaxPoolLayer(int windowSize, int stepSize) {
        if(layers.isEmpty()){
            layers.add(new MaxPool(stepSize, windowSize, 1, inRow, inCol)); 
        }else{
            Layer prevLayer = layers.get(layers.size()-1);
            layers.add(new MaxPool(stepSize, windowSize, prevLayer.getOutputLen(), prevLayer.getOutputRow(), prevLayer.getOutputCol())) ;
        }
    }
    public void addFullyConnectedLayer(int outputLen, double LR, long SEED) {
        if(layers.isEmpty()){
            layers.add(new FullyConnectedLayer( inRow*inCol,  outputLen, SEED, LR)) ; 
        }else{
            Layer prevLayer = layers.get(layers.size()-1);
            layers.add(new FullyConnectedLayer( prevLayer.getOutputElements(),  outputLen, SEED, LR)) ;
        }
    }

    public NeuralNetwork build() {
        this.net = new NeuralNetwork(layers, SF);
        return this.net;
    }
}   