import src.DataReader;
import src.Image;
import Nets.NeuralNetwork;
import Nets.NetBuilder;

import java.util.List;
import java.util.Collections;

public class Main {
    public static void main(String[] args) {

        long SEED = 42;
        System.out.println("loading data... ");
        List<Image> imagesTrain = new DataReader().readData("data/mnist_train.csv");
        List<Image> imagesTest = new DataReader().readData("data/mnist_test.csv");

        System.out.println("data loaded... ");
        System.out.println("Train Size : " + imagesTrain.size());
        System.out.println("Test Size : " + imagesTest.size());

        NetBuilder builder = new NetBuilder(28, 28, 256 * 100);
        builder.addConvulutionLayer(8, 5, 1, 0.1, SEED);
        builder.addMaxPoolLayer(3, 2);
        builder.addFullyConnectedLayer(10, 0.1, SEED);

        NeuralNetwork net = builder.build();

        float rate = net.test(imagesTest);
        System.out.println("Accuracy : [PRE TRAINING ACCURACY] : " + rate);

        int EPOCH = 3 ;

        
        System.out.println("*****************  [POST TRAINING ACCURACY] *************** ");

        for (int i = 0; i < EPOCH; i++) {
            Collections.shuffle(imagesTrain);
            net.train(imagesTrain);
            rate = net.test(imagesTest);
            System.out.print("EPOCH : " + (i+1) + " | ");
            System.out.println("Accuracy : [POST TRAINING ACCURACY] : " + rate);
        }
    }
}
