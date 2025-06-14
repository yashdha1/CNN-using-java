package src;

public class Image {
    private double[][] data;
    private int label;

    // getter
    public double[][] getData() {
        return data;
    }

    public int getLabel() {
        return label;
    }

    public Image(double[][] data, int label) {
        this.data = data;
        this.label = label;
    }

    public String visual() {
        StringBuilder sb = new StringBuilder();
        sb.append("Label: ").append(this.label).append("\n\n");

        for (int i = 0; i < data.length; i++) {
            for (int j = 0; j < data[0].length; j++) {
                int val = (int) data[i][j];

                // Use dot for zero, padded numbers otherwise
                if (val == 0) {
                    sb.append(" . ");
                } else {
                    sb.append(String.format("%2d ", val));
                }
            }
            sb.append("\n");
        }

        return sb.toString();
    }

}