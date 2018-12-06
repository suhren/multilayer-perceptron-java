/**
 * A class representing one layer in the MLP
 * @author Adam
 *
 */
public class Layer {
	
	@FunctionalInterface
	interface ActivationFunction { double eval(double x); }
	
	static private ActivationFunction identity = (x) -> x;
	static private ActivationFunction binaryStep = (x) -> (x < 0) ? 0 : 1;
	static private ActivationFunction elliotSig = (x) -> x / (1 + Math.abs(x));
	static private ActivationFunction relu = (x) -> (x < 0) ? 0 : x;
	
	public double[] o;
	public double[][] w;
	public double[] b;
	
	public Layer(int wRows, int wCols) {
		this(new double[wRows][wCols], new double[wRows]);
	}
	
	public Layer(double[][] w, double[]b) {
		if (w == null || b == null)
			throw new NullPointerException("Layer parameters not defined.");
		this.w = w;
		this.b = b;
		o = new double[w.length];
	}
	
	public double[] feedForward(double[] input) {
		o = activate(vecAdd(matVecMult(w, input), b), relu);
		return o;
	}
	
	private static double[] activate(double[] a, ActivationFunction aFun) {
		for (int i = 0; i < a.length; i++)
			a[i] = aFun.eval(a[i]);
		return a;
	}
	
	private static double[] vecAdd(double[] a, double[] b) {
		if (a.length != b.length)
			throw new IllegalArgumentException("Vectors have to be of the same length.");
		
		double[] result = new double[a.length];
		
		for (int i = 0; i < a.length; i++)
			result[i] = a[i] + b[i];
		
		return result;
	}
	
	public static double[] vecSub(double[] a, double[] b) {
		if (a.length != b.length)
			throw new IllegalArgumentException("Vectors have to be of the same length.");
		
		double[] result = new double[a.length];
		
		for (int i = 0; i < a.length; i++)
			result[i] = a[i] - b[i];
		
		return result;
	}
	
	private static double[] matVecMult(double[][] matrix, double[] vector) {
		if (matrix[0].length != vector.length)
			throw new IllegalArgumentException("The number of columns in the matrix has to be equal to the number of elements in the vector.");
		
		double[] result = new double[matrix.length];
		
		for (int row = 0; row < matrix.length; row++)
			result[row] = dotProduct(matrix[row], vector);
		
		return result;
		
	}
	
	public static int dotProduct(double[] a, double[] b) {
		if (a.length != b.length)
			throw new IllegalArgumentException("Vectors have to be of the same length.");
		
		int result = 0;
		for (int i = 0; i < a.length; i++)
			result += a[i] * b[i];
		return result;
	}
	
	@Override
	public String toString() {
		StringBuilder builder = new StringBuilder();
		for (int row = 0; row < w.length; row++) {
			for (int col = 0; col < w[0].length; col++)
				builder.append(w[row][col] + " ");
			builder.append(b[row] + " ");
			builder.append(o[row] + "\n");
		}
			
		return builder.toString();
	}
}