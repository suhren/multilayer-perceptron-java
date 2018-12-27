package com.mlp.math;

public class Utils {
	// List of activation functions:
	// https://en.wikipedia.org/wiki/Activation_function
	
	// Identity
	public static ActivationFunction identity = new ActivationFunction(
			(x) -> x,
			(x) -> 1);
	// Binary step
	public static ActivationFunction binaryStep = new ActivationFunction(
			(x) -> (x < 0) ? 0 : 1,
			(x) -> 0);
	// Logistic (a.k.a. Sigmoid or Soft step)
	public static ActivationFunction logistic = new ActivationFunction(
			(x) -> 1.0/(1 + Math.exp(-x)),
			(x) -> 1.0/(1 + Math.exp(-x)) * (1 - 1.0/(1 + Math.exp(-x))));
	// TanH
	public static ActivationFunction tanH = new ActivationFunction(
			(x) -> Math.tanh(x),
			(x) -> 1 - Math.pow(Math.tanh(x), 2));
	// ArcTan
	public static ActivationFunction arcTan = new ActivationFunction(
			(x) -> Math.atan(x),
			(x) -> 1.0 / (x*x + 1));
	// ElliotSig SoftSign	
	public static ActivationFunction elliotSig = new ActivationFunction(
			(x) -> x / (1 + Math.abs(x)),
			(x) -> 1.0 / Math.pow(1 + Math.abs(x), 2));
	// Rectified Linear Unit (ReLU)
	public static ActivationFunction relu = new ActivationFunction(
			(x) -> (x < 0) ? 0 : x,
			(x) -> (x < 0) ? 0 : 1);
	// Leaky Rectified Linear Unit (Leaky ReLU)
	public static ActivationFunction relu_leaky = new ActivationFunction(
			(x) -> (x < 0) ? 0.01 * x : x,
			(x) -> (x < 0) ? 0.01 : 1);
	// SoftPlus
	public static ActivationFunction softPlus = new ActivationFunction(
			(x) -> Math.log(1 + Math.exp(x)),
			(x) -> 1.0 / (1 + Math.exp(-x)));
	// Sinusoid
	public static ActivationFunction sinusoid = new ActivationFunction(
			(x) -> Math.sin(x),
			(x) -> Math.cos(x));
	// Sinc
	public static ActivationFunction sinc = new ActivationFunction(
			(x) -> (x == 0) ? 0 : Math.sin(x) / x,
			(x) -> (x == 0) ? 0 : Math.cos(x) / x - Math.sin(x) / x*x);
	// Gaussian
	public static ActivationFunction gaussian = new ActivationFunction(
			(x) -> Math.exp(-x*x),
			(x) -> -2*x*Math.exp(-x*x));
	
	public static double[] vecAdd(double[] a, double[] b) {
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
	
	public static double[] matVecMult(double[][] matrix, double[] vector) {
		if (matrix[0].length != vector.length)
			throw new IllegalArgumentException("The number of columns in the matrix has to be equal to the number of elements in the vector.");
		
		double[] result = new double[matrix.length];
		
		for (int row = 0; row < matrix.length; row++)
			result[row] = dotProduct(matrix[row], vector);
		
		return result;
		
	}
	
	public static double dotProduct(double[] a, double[] b) {
		if (a.length != b.length)
			throw new IllegalArgumentException("Vectors have to be of the same length.");
		
		double result = 0.0;
		for (int i = 0; i < a.length; i++)
			result += a[i] * b[i];
		return result;
	}
}
