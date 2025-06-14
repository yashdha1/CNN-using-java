package Layers;

import src.MatrixUtils;

import java.util.List;
import java.util.ArrayList;
import java.util.Random;

public class ConvolutionLayer extends Layer {

    private List<double[][]> filters;
    private List<double[][]> lastInput;
    private int filterSize;
    private int stepSize;
    private long SEED;
    private double LR;

    private int Len;
    private int Row;
    private int Col;

    public ConvolutionLayer(int filterSize, int stepSize, int len, int row, int col, long SEED, int numFilters,
            double LR) {
        this.filterSize = filterSize;
        this.stepSize = stepSize;
        this.Len = len;
        this.Row = row;
        this.Col = col;
        this.SEED = SEED;
        this.LR = LR;

        generateRandomFilters(numFilters);
    }

    private void generateRandomFilters(int numFilters) {
        List<double[][]> filters = new ArrayList<>();
        Random rand = new Random(SEED);

        for (int n = 0; n < numFilters; n++) {
            double[][] filter = new double[filterSize][filterSize];

            for (int i = 0; i < filterSize; i++) {
                for (int j = 0; j < filterSize; j++) {
                    filter[i][j] = rand.nextGaussian();
                }
            }
            filters.add(filter);
        }

        this.filters = filters;

    }

    public List<double[][]> ConvulutionForwardPass(List<double[][]> input) {
        List<double[][]> output = new ArrayList<>();

        lastInput = input; // for backpropogation :->

        for (int n = 0; n < input.size(); n++) {
            for (double[][] filter : filters) {
                output.add(convolve(input.get(n), filter, stepSize));
            }
        }
        return output;
    };

    public double[][] convolve(double[][] input, double[][] filter, int stepSize) {
        int outputRow = (input.length - filter.length) / stepSize + 1;
        int outputCol = (input[0].length - filter[0].length) / stepSize + 1;

        int inRow = input.length;
        int inCol = input[0].length;

        int fRow = filter.length;
        int fCol = filter[0].length;

        double[][] output = new double[outputRow][outputCol];

        int outRow = 0;
        int outCol;

        for (int i = 0; i <= (inRow - fRow); i += stepSize) {
            outCol = 0;
            for (int j = 0; j <= (inCol - fCol); j += stepSize) {

                double sum = 0.0;
                // apply the filter around the position
                for (int x = 0; x < fRow; x++) {
                    for (int y = 0; y < fCol; y++) {
                        double sol = input[i + x][j + y] * filter[x][y];
                        sum += sol;
                    }
                }

                output[outRow][outCol] = sum;
                outCol++;
            }
            outRow++;
        }

        return output;
    }

    @Override
    public double[] getOutput(List<double[][]> input) {
        List<double[][]> output = ConvulutionForwardPass(input);
        return _nextLayer.getOutput(output);
    }

    @Override
    public double[] getOutput(double[] input) {
        List<double[][]> matrix = VectorToMatrix(input, Len, Row, Col);
        return getOutput(matrix);
    }

    @Override
    public void BackPropogation(List<double[][]> dldo) {
        List<double[][]> filtersDel = new ArrayList<>();
        List<double[][]> dldoPrevLayer = new ArrayList<>();  
        for (int f = 0; f < filters.size(); f++) {
            filtersDel.add(new double[filterSize][filterSize]);
        }

        for (int i = 0; i < lastInput.size(); i++) {
            double[][] errorForInput = new double[Row][Col] ; 
            for (int f = 0; f < filters.size(); f++) {
                double[][] filter = filters.get(f);
                double[][] error = dldo.get(i * filters.size() + f);

                double[][] spcaedError = spaceArray(error);
                double[][] dldf = convolve(lastInput.get(i), spcaedError, 1);

                double[][] delta = MatrixUtils.multiply(dldf, LR * -1);
                double[][] newTotalDelta = MatrixUtils.add(filtersDel.get(f), delta);

                filtersDel.set(f, newTotalDelta);

                double[][] flippedError = flipH(flipV(spcaedError));
                

                // fulll convulution ERROR 
                errorForInput = MatrixUtils.add(errorForInput, fullConvolve(filter, flippedError)) ;
            }

            dldoPrevLayer.add(errorForInput) ;
        }

        for (int f = 0; f < filters.size(); f++) {
            double[][] modified = MatrixUtils.add(filters.get(f), filtersDel.get(f));
            filters.set(f, modified); // filters are now learning
        }

        if(_prevLayer!=null){
            _prevLayer.BackPropogation(dldoPrevLayer); 
        }
    }

    @Override
    public void BackPropogation(double[] input) {
        List<double[][]> matrix = VectorToMatrix(input, Len, Row, Col); 
        BackPropogation(matrix);
    }

    public double[][] fullConvolve(double[][] input, double[][] filter) {
        int outputRow = (input.length + filter.length) + 1;
        int outputCol = (input[0].length + filter[0].length) + 1;

        int inRow = input.length;
        int inCol = input[0].length;

        int fRow = filter.length;
        int fCol = filter[0].length;

        double[][] output = new double[outputRow][outputCol];

        int outRow = 0;
        int outCol;

        for (int i = -fRow + 1; i < inRow; i++) {
            outCol = 0;
            for (int j = -fCol + 1; j < inCol; j++) {

                double sum = 0.0;
                // apply the filter around the position
                for (int x = 0; x < fRow; x++) {
                    for (int y = 0; y < fCol; y++) {
                        int ir = i + x;
                        int jc = j + y;
                        if (ir >= 0 && ir < inRow && jc >= 0 && jc < inCol) {
                            double sol = input[ir][jc] * filter[x][y];
                            sum += sol;
                        }
                    }
                }

                output[outRow][outCol] = sum;
                outCol++;
            }
            outRow++;
        }

        return output;
    }

    public double[][] flipH(double[][] mat) {
        double[][] flipped = new double[mat.length][mat[0].length];
        for (int i = 0; i < mat.length; i++) {
            for (int j = 0; j < mat[0].length; j++) {
                flipped[mat.length - i - 1][j] = mat[i][j];
            }
        }
        return flipped;
    }
    public double[][] flipV(double[][] mat) {
        double[][] flipped = new double[mat.length][mat[0].length];
        for (int i = 0; i < mat.length; i++) {
            for (int j = 0; j < mat[0].length; j++) {
                flipped[i][mat[0].length - j - 1] = mat[i][j];
            }
        }
        return flipped;
    }

    public double[][] spaceArray(double[][] input) {
        if (stepSize == 1) {
            return input;
        }

        int outRows = (input.length - 1) / stepSize + 1;
        int outCols = (input[0].length - 1) / stepSize + 1;
        double[][] output = new double[outRows][outCols];

        for (int r = 0; r < outRows; r++) {
            for (int c = 0; c < outCols; c++) {
                output[r + stepSize][c + stepSize] = input[r][c];
            }
        }
        return output;
    }
    @Override
    public int getOutputLen() {
        return filters.size() * Len;
    }
    @Override
    public int getOutputRow() {
        return (Row - filterSize) / stepSize + 1;
    }
    @Override
    public int getOutputCol() {
        return (Col - filterSize) / stepSize + 1;
    }
    @Override
    public int getOutputElements() {
        return getOutputLen() * getOutputRow() * getOutputCol();
    }

}