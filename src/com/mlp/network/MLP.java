package com.mlp.network;

import com.mlp.math.ActivationFunction;
import com.mlp.math.Utils;
import com.mlp.program.Main;

/**
 * A class representing a Multilayer Perceptron.
 * @author Adam
 *
 */
public class MLP {
	private Layer[] layers;
	private Layer outputLayer;
	private double cost;
	// Learning rate
	private double eta = 0.01;
	
	private double[] input;
	
	public MLP(int inputSize, int[] outputSizes, ActivationFunction aFun, double eta) {
		layers = new Layer[outputSizes.length];
		int inSize = inputSize;
		for (int i = 0; i < outputSizes.length; i++) {
			layers[i] = new Layer(outputSizes[i], inSize, aFun);
			layers[i].randomize(); 
			inSize = outputSizes[i];
		}
		
		this.outputLayer = layers[layers.length - 1];
		input = new double[inputSize];
		this.eta = eta;
	}
	
	public MLP(Layer[] layers) {
		if (layers == null)
			throw new NullPointerException("No layers specified.");
		if (layers.length < 2)
			throw new IllegalArgumentException("The MLP needs to have at least 2 layers.");
		
		// Copy over the layers for the immutable property
		this.layers = layers.clone();
		this.outputLayer = layers[layers.length - 1];
		input = new double[layers[0].w[0].length];
	}
	
	public void train(TrainingSet trainingSet) {
		for (TrainingEntry e : trainingSet.getTrainingEntries()) {
			feedThrough(e.getInput(), e.getExpectedOutput());
			backPropagate(e.getExpectedOutput());
		}
	}
	
	/**
	 * @param The input vector of the MLP.
	 * @param The expected output vector of the MLP.
	 * @return The output of the MLP.
	 */
	public double[] feedThrough(double[] input, double[] expectedOutput) {
		if (outputLayer.out.length != expectedOutput.length)
			throw new IllegalArgumentException("The dimensions of the expected output vector has to be equal to the dimensions of the output vector in the output layer.");
		
		this.input = input.clone();
		double[] current = input.clone();
		
		for (Layer l : layers)
			current = l.feedForward(current);

		cost = calcCost(expectedOutput, outputLayer.out);
		
		return current;
	}
	
	/**
	 * Perform back propagation updating the weights of the layers using the latest output of MLP.
	 */
	public void backPropagate(double[] expectedOutput) {
		// Output layer
		for (int row = 0; row < outputLayer.w.length; row++) {
//			double dEdO = 2 * 1.0 / outputLayer.w.length * (outputLayer.out[row] - expectedOutput[row]);
			double dEdO = 2 * (outputLayer.out[row] - expectedOutput[row]);
			double dOdN = outputLayer.aFun.evalDerivate(outputLayer.netIn[row]);
			outputLayer.dEdN[row] = dEdO * dOdN;
			double dEdB = dEdO * dOdN * 1;
			outputLayer.bNew[row] = outputLayer.b[row] - eta * dEdB;
			for (int col = 0; col < outputLayer.w[0].length; col++) {
				double dNdW = outputLayer.in[col];
				double dEdW = dEdO * dOdN * dNdW;
				outputLayer.wNew[row][col] = outputLayer.w[row][col] - eta * dEdW;
			}
		}
		
		// Hidden layers
		for (int i = layers.length - 2; i >= 0; i--) {
			Layer l = layers[i];
			Layer nl = layers[i + 1];
			for (int row = 0; row < l.w.length; row++) {
				double dOdN = l.aFun.evalDerivate(l.netIn[row]);
				double dEdO = 0;
//				for (int j = 0; j < nl.w.length; j++)
//					for (int k = 0; k < nl.w[0].length; k++)
//						dEdO += nl.dEdN[j] * nl.w[row][k];
				for (int j = 0; j < nl.w.length; j++)
					dEdO += nl.dEdN[j] * nl.w[j][row];
				
				double dEdN = dEdO * dOdN;
				double dEdB = dEdN * 1;
				l.bNew[row] = l.b[row] - eta * dEdB;
				l.dEdN[row] = dEdN;
				for (int col = 0; col < l.w[0].length; col++) { 
					double dNdW = l.in[col];
					double dEdW = dEdN * dNdW;
					l.wNew[row][col] = l.w[row][col] - eta * dEdW;
				}
			}
		}
		
		for (Layer l : layers) {
			l.w = l.wNew.clone();
			l.b = l.bNew.clone();
		}
	}
	
	/**
	 * @param The first vector a.
	 * @param The second vector b.
	 * @return The cost of the two vectors.
	 */
	private double calcCost(double[] a, double[] b) {
		double[] d = Utils.vecSub(a, b);
//		return Math.sqrt(Utils.dotProduct(d, d));
//		return 1.0 / d.length * Utils.dotProduct(d, d);
		return Utils.dotProduct(d, d);
	}
	
	/**
	 * @return The cost of the last output of the MLP.
	 */
	public double getCost() { return cost; }
	/**
	 * @return The layers of the MLP.
	 */
	public Layer[] getLayers() { return layers; }
	
	public double[] getInputLayer() { return input; }
	
	@Override
	public String toString() {
		int dSize = 7;
		int sep = 4;
		int totalWidth = 20;
		int totalHeight = 0;
		
		totalWidth += dSize;	// For the column in input
		totalWidth += 2;		// 2 for vertical bars around input
		
		if (input.length > totalHeight)
			totalHeight = input.length;
		
		for (Layer l : layers) {							
			totalWidth += sep;								// For separation of layers
			totalWidth += 1 + l.w[0].length * (dSize + 1);	// For each column and vertical bars in w
			totalWidth += 1;								// For the space between w and b
			totalWidth += (1 + dSize + 1);					// For the column and vertical bars in b
			totalWidth += 1;								// For the space between b and s
			totalWidth += (1 + dSize + 1);					// For the column and vertical bars in s
			
			if (l.w.length > totalHeight)
				totalHeight = l.w.length;
		}
		
		CharMap map = new CharMap(totalWidth, totalHeight);
		
		int x = 0;
		int y = (totalHeight - input.length) / 2;		// Center layer in y
		
		for (int i = 0; i < input.length; i++)
			map.add(String.format("|%2.2e|", input[i]), x, y + i);
		
		x += (1 + dSize + 1 + sep + 1);
		
		for (Layer l : layers) {
			y = (totalHeight - l.w.length) / 2;
			for (int row = 0; row < l.w.length; row++) {
				String s = "|";
				s += String.format("%2.2f", l.w[row][0]);
				for (int col = 1; col < l.w[0].length; col++)
					s += String.format(" %2.2e", l.w[row][col]);
				s += "|";
				map.add(s, x, y + row);
			}
			
			x += (1 + l.w[0].length * (dSize + 1) + 1);
			
			y = (totalHeight - l.b.length) / 2;
			for (int row = 0; row < l.b.length; row++)
				map.add(String.format("|%2.2e|", l.b[row]), x, y + row);

			x += (1 + dSize + 2);
			
			y = (totalHeight - l.out.length) / 2;
			for (int row = 0; row < l.out.length; row++)
				map.add(String.format("|%2.2e|", l.out[row]), x, y + row);
			
			x += (1 + dSize + 1 + sep);
		}
		
		return(map.toString());
	}
	
	class CharMap {
		public char[] map;
		public int n, w, h;
		
		public CharMap(int w, int h) {
			this.w = w;
			this.h= h;
			this.n= w * h;
			this.map = new char[n];
		}
		public void add(double d, int x, int y) {
			add(String.format("%.2f", d), x, y);
		}
		private void add(String s, int x, int y) {
			for (int i = 0; i < s.length(); i++)
				add(s.charAt(i), x + i, y);
		}
		private void add(char c, int x, int y) {
			map[x + y * w] = c;
		}
		private char get(int x, int y) {
			return map[x + y * w];
		}
		@Override
		public String toString() {
			StringBuilder s = new StringBuilder();
			for (int y = 0; y < h; y++) {
				for (int x = 0; x < w; x++)
					s.append(get(x, y));
				s.append("\n");
			}
			return s.toString();
		}
	}
}