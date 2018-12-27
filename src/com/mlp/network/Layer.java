package com.mlp.network;

import java.util.Random;

import com.mlp.math.ActivationFunction;
import com.mlp.math.Utils;

/**
 * A class representing one layer in the MLP
 * @author Adam
 *
 */
public class Layer {
	// The input From the previous layer
	public double[] in;
	// The weights
	public double[][] w;
	// The new weights
	public double[][] wNew;
	// The biases
	public double[] b;
	// The biases
	public double[] bNew;
	// The net input
	public double[] netIn;
	// The activation function used by the layer
	public ActivationFunction aFun;
	// The activated output
	public double[] out;
	// The error in the output layer in regards to the input
	public double[] dEdN;
	
	public Layer(int wRows, int wCols) {
		this(new double[wRows][wCols], new double[wRows], Utils.identity);
	}
	
	public Layer(int wRows, int wCols, ActivationFunction aFun) {
		this(new double[wRows][wCols], new double[wRows], aFun);
	}
	
	public Layer(double[][] w, double[]b, ActivationFunction aFun) {
		if (w == null || b == null)
			throw new NullPointerException("Layer parameters not defined.");
		this.w = w;
		this.wNew = new double[w.length][w[0].length];
		this.b = b;
		this.bNew = new double[b.length];
		this.in = new double[w[0].length];
		this.netIn = new double[w.length];
		this.aFun = aFun;
		this.out = new double[w.length];
		this.dEdN = new double[w.length];
	}
	
	public void randomize() {
		Random rand = new Random();
		for (int row = 0; row < w.length; row++) {
			b[row] = rand.nextDouble();
			for (int col = 0; col < w[0].length; col++)
				w[row][col] = rand.nextDouble();
		}
	}
	
	public double[] feedForward(double[] input) {
		in = input.clone();
		netIn = Utils.vecAdd(Utils.matVecMult(w, in), b);
		out = activate(netIn, aFun);
		return out;
	}
	
	private static double[] activate(double[] a, ActivationFunction aFun) {
		double[] result = new double[a.length];
		for (int i = 0; i < a.length; i++)
			result[i] = aFun.eval(a[i]);
		return result;
	}
	
	@Override
	public String toString() {
		StringBuilder builder = new StringBuilder();
		for (int row = 0; row < w.length; row++) {
			for (int col = 0; col < w[0].length; col++)
				builder.append(w[row][col] + " ");
			builder.append(b[row] + " ");
			builder.append(out[row] + "\n");
		}
			
		return builder.toString();
	}
}