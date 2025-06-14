package Layers;

import java.util.List;
import java.util.ArrayList;

public abstract class Layer {
    protected Layer _nextLayer;
    protected Layer _prevLayer;
    
    //  Getters and setters for the Layers
    public void setNextLayer(Layer nextLayer) {
        this._nextLayer = nextLayer;
    }
    public void setPrevLayer(Layer prevLayer) {
        this._prevLayer = prevLayer;
    }

    public Layer getNextLayer() {
        return this._nextLayer;
    }
    public Layer getPrevLayer() {
        return this._prevLayer;
    }

    // CONNECTED LAYER 
    public abstract double[] getOutput(List<double[][]> input);
    public abstract void BackPropogation(List<double[][]> input);

    // MAXPOOL LAYER 
    public abstract double[] getOutput(double[] input);
    public abstract void BackPropogation(double[] input); 

    // METRICS 
    public abstract int getOutputLen();
    public abstract int getOutputRow();
    public abstract int getOutputCol();
    public abstract int getOutputElements();


    public double[] MatrixToVector(List<double[][]> input) {
        int len = input.size();
        int r = input.get(0).length;
        int c = input.get(0)[0].length;

        double[] vector = new double[len * r * c];
        int id = 0;
        for (int i = 0; i < len; i++) {
            for (int j = 0; j < r; j++) {
                for (int k = 0; k < c; k++) {
                    vector[id] = input.get(i)[j][k];
                }
            }
        }
        return vector;
    }

    public List<double[][]> VectorToMatrix(double[] input, int len, int r, int c) {
        List<double[][]> output = new ArrayList<>();
        int id = 0;
        for (int i = 0; i < len; i++) {
            double[][] temp = new double[r][c];
            for (int j = 0; j < r; j++) {
                for (int k = 0; k < c; k++) {
                    temp[j][k] = input[id++];
                }
            }
            output.add(temp);
        }
        return output;
    }

}
