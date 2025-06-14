package Layers;

import java.util.List;
import java.util.ArrayList;

public class MaxPool extends Layer {

    private int _stepSize;
    private int _winSize;

    private int _len;
    private int _row;
    private int _col;

    List<int[][]> lastMaxRow;
    List<int[][]> lastMaxCol;

    public MaxPool(int stepSize, int winSize, int len, int row, int col) {
        this._stepSize = stepSize;
        this._winSize = winSize;
        this._len = len;
        this._row = row;
        this._col = col;
    }

    public List<double[][]> maxPoolForwardPass(List<double[][]> input) {
        List<double[][]> output = new ArrayList<>();

        lastMaxRow = new ArrayList<>();
        lastMaxCol = new ArrayList<>();

        for (int i = 0; i < input.size(); i++) {
            output.add(pool(input.get(i)));
        }
        return output;
    }

    public double[][] pool(double[][] input) {
        double[][] output = new double[getOutputRow()][getOutputCol()];
        int[][] maxRow = new int[getOutputRow()][getOutputCol()];
        int[][] maxCol = new int[getOutputRow()][getOutputCol()];

        for (int i = 0; i < getOutputRow(); i += _stepSize) {
            for (int j = 0; j < getOutputCol(); j += _stepSize) {

                double max = 0.0;
                for (int x = 0; x < _winSize; x++) {
                    for (int y = 0; y < _winSize; y++) {
                        if (input[i + x][j + y] > max) {
                            max = input[i + x][j + y];
                            maxRow[i][j] = i + x;
                            maxCol[i][j] = j + y;
                        }
                    }
                }
                // we will have the max in the pool with this fuction
                output[i][j] = max;
            }
        }

        lastMaxRow.add(maxRow);
        lastMaxCol.add(maxCol);
        return output;
    }

    @Override
    public double[] getOutput(double[] input) {
        List<double[][]> matrixList = VectorToMatrix(input, _len, _row, _col);
        return getOutput(matrixList);
    }

    @Override
    public double[] getOutput(List<double[][]> input) {
        List<double[][]> outputPool = maxPoolForwardPass(input);
        // System.out.println();

        // ERROR
        // return _nextLayer.getOutput(outputPool);

        // GPT SOLOUTION
        if (_nextLayer != null) {
            return _nextLayer.getOutput(outputPool);
        }
        return MatrixToVector(outputPool); // Implement this if needed
    }

    @Override
    public void BackPropogation(double[] input) {
        List<double[][]> matrixList = VectorToMatrix(input, getOutputLen(), getOutputRow(), getOutputCol());
        BackPropogation(matrixList);

    }

    @Override
    public void BackPropogation(List<double[][]> dldo) {
        List<double[][]> dxdl = new ArrayList<>();
        int l = 0;
        for (double[][] input : dldo) {
            double[][] error = new double[_row][_col];
            for (int i = 0; i < getOutputRow(); i++) {
                for (int j = 0; j < getOutputCol(); j++) {
                    int max_i = lastMaxRow.get(l)[i][j];
                    int max_j = lastMaxCol.get(l)[i][j];
                    if (max_i != -1)
                        error[max_i][max_j] += input[i][j];
                }
            }
            dxdl.add(error);
            l++;
        }

        if (_prevLayer != null) {
            _prevLayer.BackPropogation(dxdl);
        }
    }

    @Override
    public int getOutputLen() {
        return _len;
    }

    @Override
    public int getOutputRow() {
        return (_row - _winSize) / _stepSize + 1;
    }

    @Override
    public int getOutputCol() {
        return (_col - _winSize) / _stepSize + 1;
    }

    @Override
    public int getOutputElements() {
        return _len * getOutputRow() * getOutputCol();
    }

}
