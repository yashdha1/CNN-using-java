package src;

import java.io.BufferedReader;
import java.io.FileReader;
import java.util.ArrayList;
import java.util.List;

public class DataReader {
    // 28 * 28 pixel image
    private final int row = 28;
    private final int col = 28;

    public List<Image> readData(String path){
    List<Image> images = new ArrayList<>(); 
    try (BufferedReader reader = new BufferedReader(new FileReader(path))) {
        String line = reader.readLine();
        while (line != null) {
            String[] tokens = line.split(","); 
            double[][] data = new double[row][col]; 
            int label = Integer.parseInt(tokens[0]);

            int k = 1; 
            for(int i = 0; i < row; i++){
                for(int j = 0; j < col; j++){
                    data[i][j] = (double) Integer.parseInt(tokens[k]);
                    k++; 
                }
            }

            images.add(new Image(data, label));
            line = reader.readLine(); 
        }
    } catch (Exception e) {
        e.printStackTrace(); // Better for debugging
    }

    return images; 
}

}
