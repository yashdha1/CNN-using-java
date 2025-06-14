package Nets;

import Layers.*;
import src.MatrixUtils;
import src.Image;

import java.util.List;
import java.util.ArrayList;

public class NeuralNetwork {
    List<Layer> layers;
    double SF; // scale factor to normalize the data

    public NeuralNetwork(List<Layer> layers, double SF) {
        this.layers = layers;
        this.SF = SF;
        linkLayers();
    }

    private void linkLayers() {
        if (layers.size() <= 1)
            return;

        for (int i = 0; i < layers.size() - 1; i++) {
            if (i == 0) {
                layers.get(i).setNextLayer(layers.get(i + 1));
            } else if (i == layers.size() - 2) {
                layers.get(i).setPrevLayer(layers.get(i - 1));
            } else {
                layers.get(i).setNextLayer(layers.get(i + 1));
                layers.get(i).setPrevLayer(layers.get(i - 1));
            }
        }
    }

    public double[] getErrors(double[] netOut, int corr) {
        int numClasses = netOut.length;
        double[] expected = new double[numClasses];
        expected[corr] = 1;
        return MatrixUtils.add(netOut, MatrixUtils.multiply(expected, -1));
    }

    public int getMaxIndex(double[] arr) {
        double max = 0.0;
        int index = 0;

        for (int i = 0; i < arr.length; i++) {
            if (arr[i] > max) {
                max = arr[i];
                index = i;
            }
        }
        return index;

        // works only for the MNIST DATASET
        // TO MAKE THIS UNIVERSAL :
        // MAke a class that will take the array and return the index
        // and find way to map the index to the class
    }

    public int guess(Image image) {
        List<double[][]> inList = new ArrayList<>();
        inList.add(MatrixUtils.multiply(image.getData(), 1 / SF));

        double[] out = layers.get(0).getOutput(inList);

        return getMaxIndex(out);
    }

    public float test(List<Image> images) {
        int correct = 0;

        // for every image:->
        for (Image image : images) {
            int guess = guess(image);
            if (guess == image.getLabel())
                correct++;
        }
        return correct / (float) images.size();
    }

    public void train(List<Image> images) {

        for (Image image : images) {
            List<double[][]> inList = new ArrayList<>();
            inList.add(MatrixUtils.multiply(image.getData(), 1 / SF));
            double[] out = layers.get(0).getOutput(inList);
            double[] dldo = getErrors(out, image.getLabel());

            layers.get((layers.size() - 1)).BackPropogation(dldo);
        }
    }
}
