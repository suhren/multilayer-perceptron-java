package com.mlp.network;

import java.util.Random;

import com.mlp.math.AFun;
import com.mlp.math.Matrix;
import com.mlp.math.Utils;
import com.mlp.math.Vector;

/**
 * A class representing one layer in the MLP
 * @author Adam
 *
 */
public class Layer {
	Vector in; 	// The input From the previous layer
	Matrix w; 	// The weights
	Vector b; 		// The biases
	Vector z;		// The net input
	AFun aFun; // The activation function
	Vector out; 	// The activated output
	
	Vector dEdZ; 	// The error in the output layer in regards to the input
	Matrix wNew; // The new weights
	Vector bNew; 	// The new biases
	
	public Layer(int wRows, int wCols) {
		this(new Matrix(wRows, wCols), new Vector(wRows), Utils.AFunLibrary.IDENTITY.aFun());
	}
	public Layer(int wRows, int wCols, AFun aFun) {
		this(new Matrix(wRows, wCols), new Vector(wRows), aFun);
	}
	public Layer(Matrix w, Vector b, AFun aFun) {
		if (w == null || b == null)
			throw new NullPointerException("Layer parameters not defined.");
		this.w = w;
		this.wNew = new Matrix(w.nRow(), w.nCol());
		this.b = b;
		this.bNew = new Vector(b.size());
		this.in = new Vector(w.nCol());
		this.z = new Vector(w.nRow());
		this.aFun = aFun;
		this.out = new Vector(w.nRow());
		this.dEdZ = new Vector(w.nRow());
	}
	
	public void randomize() {
		Random rand = new Random();
		for (int row = 0; row < w.nRow(); row++) {
			b.set(row, rand.nextDouble());
			for (int col = 0; col < w.nCol(); col++)
				w.set(row, col, rand.nextDouble());
		}
	}
	
	public Vector feedForward(Vector input) {
		in = input.clone();
		z = w.multiply(in).add(b);
		out = aFun.eval(z);
		return out;
	}
	
	public Matrix getWeights() {
		return w;
	}
	
	public Vector getBiases() {
		return b;
	}
	
	public AFun getAFun() {
		return aFun;
	}
	
	@Override
	public String toString() {
		StringBuilder builder = new StringBuilder();
		for (int row = 0; row < w.nRow(); row++) {
			for (int col = 0; col < w.nCol(); col++)
				builder.append(w.get(row, col) + " ");
			builder.append(b.get(row) + " ");
			builder.append(out.get(row) + "\n");
		}
		return builder.toString();
	}
}