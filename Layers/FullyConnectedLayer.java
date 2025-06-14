package Layers;

import java.util.Random;
import java.util.List;

public class FullyConnectedLayer extends Layer {

    private double[][] _weights;
    private int _inputLen;
    private int _outputLen;
    private long SEED;

    // derivatives tracking:->
    private double[] lastZ;
    private double[] lastX;
    private final double leak = 0.01;
    private double LR;

    public FullyConnectedLayer(int inputLen, int outputLen, long SEED, double LR) {
        this._inputLen = inputLen;
        this._outputLen = outputLen;
        this.SEED = SEED;
        this.LR = LR;
        _weights = new double[this._inputLen][this._outputLen];

        setRandomWeight();
    }

    // FORWARD PASS
    public double[] ForwardPassFullyConnected(double[] input) {

        lastX = input;
        double[] o1 = new double[_outputLen];
        double[] output = new double[_outputLen];

        // out = in * w
        for (int i = 0; i < _inputLen; i++) {
            for (int j = 0; j < _outputLen; j++) {
                o1[j] += input[i] * _weights[i][j];
            }
        }

        // ReLU
        lastZ = o1; // keep track of z before the RELU:->
        for (int i = 0; i < _outputLen; i++) {
            for (int j = 0; j < _outputLen; j++) {
                output[i] = ReLU(o1[j]);
            }
        }

        return output;
    }

    @Override
    public double[] getOutput(List<double[][]> input) {
        double[] vector = MatrixToVector(input);
        return getOutput(vector);
    }

    @Override
    public double[] getOutput(double[] input) {
        double[] output = ForwardPassFullyConnected(input);

        if (_nextLayer != null) {
            return _nextLayer.getOutput(output);
        } else {
            // last layer condition:
            return output;
        }
    }

    @Override
    public void BackPropogation(double[] dLdO) {

        double[] dldx = new double[_inputLen];
        // chain method : dO/dz , dzdx, dL/dw
        double dodz;
        double dzdw;
        double dldw;
        double dzdx;

        for (int i = 0; i < _inputLen; i++) {
            double dldx_sum = 0;

            for (int j = 0; j < _outputLen; j++) {

                dodz = DerivativeReLU(lastZ[j]);
                dzdw = lastX[i];
                dzdx = _weights[i][j];

                dldw = dLdO[j] * dodz * dzdw;
                _weights[i][j] -= LR * dldw;

                dldx_sum += dLdO[j] * dodz * dzdx;
            }

            dldx[i] = dldx_sum;

            if (_prevLayer != null) {
                _prevLayer.BackPropogation(dldx);
            }
        }
    }

    @Override
    public void BackPropogation(List<double[][]> input) {
        double[] vector = MatrixToVector(input);
        BackPropogation(vector);
    }

    @Override
    public int getOutputLen() {
        return 0;
    }

    @Override
    public int getOutputCol() {
        return 0;
    }

    @Override
    public int getOutputRow() {
        return 0;
    }

    @Override
    public int getOutputElements() {
        return _outputLen;
    }

    public void setRandomWeight() {
        Random rand = new Random(SEED);
        for (int i = 0; i < this._inputLen; i++) {
            for (int j = 0; j < this._outputLen; j++) {
                this._weights[i][j] = rand.nextGaussian();
            }
        }
    }

    public double ReLU(double input) {
        return Math.max(0, input);
    }

    public double DerivativeReLU(double input) {
        if (input <= 0)
            return leak;
        else
            return 1;
    }

}
