package src;

public class MatrixUtils {

    public static double[][] add(double[][] a, double[][] b) {
        double[][] res = new double[a.length][a[0].length]; 

        for(int i=0; i<a.length; i++) {
            for(int j=0; j<a[0].length; j++) {
                res[i][j] = a[i][j] + b[i][j]; 
            }
        }
        return res;
    }
    public static double[] add(double[] a, double[] b) {
        double[] res = new double[a.length] ; 

        for(int i=0; i<a.length; i++) { 
                res[i] = a[i] + b[i]; 
        }
        return res;
    }
    
    public static double[][] multiply(double[][] a, double scaler) {
        double[][] res = new double[a.length][a[0].length]; 

        for(int i=0; i<a.length; i++) {
            for(int j=0; j<a[0].length; j++) {
                res[i][j] = a[i][j]*scaler; 
            }
        }
        return res;
    }
    public static double[] multiply(double[] a, double scaler) {
        double[] res = new double[a.length]; 

        for(int i=0; i<a.length; i++) { 
                res[i] = a[i]*scaler;  
        }
        return res;
    }


}
