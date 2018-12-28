package com.mlp.network;

import com.mlp.math.AFun;
import com.mlp.math.Vector;

/**
 * A class representing a Multilayer Perceptron.
 * @author Adam
 *
 */
public class MLP {
	private Layer[] layers;
	private Layer outLayer;
	private double cost;
	// Learning rate
	private double eta = 0.01;
	
	private Vector input;
	
	public MLP(int inputSize, int[] outputSizes, AFun aFun, double eta) {
		layers = new Layer[outputSizes.length];
		int inSize = inputSize;
		for (int i = 0; i < outputSizes.length; i++) {
			layers[i] = new Layer(outputSizes[i], inSize, aFun);
			layers[i].randomize(); 
			inSize = outputSizes[i];
		}
		
		this.outLayer = layers[layers.length - 1];
		input = new Vector(inputSize);
		this.eta = eta;
	}
	
	public MLP(Layer[] layers) {
		if (layers == null)
			throw new NullPointerException("No layers specified.");
		if (layers.length < 2)
			throw new IllegalArgumentException("The MLP needs to have at least 2 layers.");
		
		// Copy over the layers for the immutable property
		this.layers = layers.clone();
		this.outLayer = layers[layers.length - 1];
		input = new Vector(layers[0].w.nCol());
	}
	
	public double train(TrainingSet trainingSet) {
		double sum = 0.0;
		for (TrainingEntry e : trainingSet.getTrainingEntries()) {
			feedThrough(e.getInput(), e.getExpectedOutput());
			sum += cost;
			backPropagate(e.getExpectedOutput());
		}
		return sum / trainingSet.getTrainingEntries().size();
	}
	
	/**
	 * @param The input vector of the MLP.
	 * @param The expected output vector of the MLP.
	 * @return The output of the MLP.
	 */
	public Vector feedThrough(Vector input, Vector expectedOutput) {
		if (outLayer.out.size() != expectedOutput.size())
			throw new IllegalArgumentException("The dimensions of the expected output vector has to be equal to the dimensions of the output vector in the output layer.");
		
		this.input = input.clone();
		Vector current = input.clone();
		
		for (Layer l : layers)
			current = l.feedForward(current);

		cost = calcCost(expectedOutput, outLayer.out);
		
		return current;
	}
	
	/**
	 * Perform back propagation updating the weights of the layers using the latest output of MLP.
	 */
	public void backPropagate(Vector expectedOutput) {
		// Output layer
		for (int row = 0; row < outLayer.w.nRow(); row++) {
//			double dEdO = 2 * 1.0 / outputLayer.w.length * (outputLayer.out[row] - expectedOutput[row]);
			double dEdO = 2 * (outLayer.out.get(row) - expectedOutput.get(row));
			double dOdZ = outLayer.aFun.evalDerivate(outLayer.z.get(row));
			outLayer.dEdZ.set(row, dEdO * dOdZ);
			double dEdB = dEdO * dOdZ * 1;
			outLayer.bNew.set(row, outLayer.b.get(row) - eta * dEdB);
			for (int col = 0; col < outLayer.w.nCol(); col++) {
				double dZdW = outLayer.in.get(col);
				double dEdW = dEdO * dOdZ * dZdW;
				outLayer.wNew.set(row, col, outLayer.w.get(row, col) - eta * dEdW);
			}
		}
		
		// Hidden layers
		for (int i = layers.length - 2; i >= 0; i--) {
			Layer l = layers[i];
			Layer nl = layers[i + 1];
			for (int row = 0; row < l.w.nRow(); row++) {
				double dOdZ = l.aFun.evalDerivate(l.z.get(row));
				double dEdO = 0;
//				for (int j = 0; j < nl.w.length; j++)
//					for (int k = 0; k < nl.w[0].length; k++)
//						dEdO += nl.dEdN[j] * nl.w[row][k];
				for (int j = 0; j < nl.w.nRow(); j++)
					dEdO += nl.dEdZ.get(j) * nl.w.get(j, row);
				
				double dEdZ = dEdO * dOdZ;
				double dEdB = dEdZ * 1;
				l.bNew.set(row, l.b.get(row) - eta * dEdB);
				l.dEdZ.set(row, dEdZ);
				for (int col = 0; col < l.w.nCol(); col++) { 
					double dZdW = l.in.get(col);
					double dEdW = dEdZ * dZdW;
					l.wNew.set(row, col, l.w.get(row, col) - eta * dEdW);
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
	private double calcCost(Vector a, Vector b) {
		Vector d = a.sub(b);
//		return Math.sqrt(Utils.dotProduct(d, d));
//		return 1.0 / d.length * Utils.dotProduct(d, d);
		return d.dotProduct(d);
	}
	
	/**
	 * @return The cost of the last output of the MLP.
	 */
	public double getCost() { return cost; }
	/**
	 * @return The layers of the MLP.
	 */
	public Layer[] getLayers() { return layers; }
	
	public Vector getInputLayer() {
		return input;
	}
	
	@Override
	public String toString() {
		int dSize = 7;
		int sep = 4;
		int totalWidth = 20;
		int totalHeight = 0;
		
		totalWidth += dSize;	// For the column in input
		totalWidth += 2;		// 2 for vertical bars around input
		
		if (input.size() > totalHeight)
			totalHeight = input.size();
		
		for (Layer l : layers) {							
			totalWidth += sep;								// For separation of layers
			totalWidth += 1 + l.w.nCol() * (dSize + 1);	// For each column and vertical bars in w
			totalWidth += 1;								// For the space between w and b
			totalWidth += (1 + dSize + 1);					// For the column and vertical bars in b
			totalWidth += 1;								// For the space between b and s
			totalWidth += (1 + dSize + 1);					// For the column and vertical bars in s
			
			if (l.w.nRow() > totalHeight)
				totalHeight = l.w.nRow();
		}
		
		CharMap map = new CharMap(totalWidth, totalHeight);
		
		int x = 0;
		int y = (totalHeight - input.size()) / 2;		// Center layer in y
		
		for (int i = 0; i < input.size(); i++)
			map.add(String.format("|%2.2e|", input.get(i)), x, y + i);
		
		x += (1 + dSize + 1 + sep + 1);
		
		for (Layer l : layers) {
			y = (totalHeight - l.w.nRow()) / 2;
			for (int row = 0; row < l.w.nRow(); row++) {
				String s = "|";
				s += String.format("%2.2f", l.w.get(row, 0));
				for (int col = 1; col < l.w.nCol(); col++)
					s += String.format(" %2.2e", l.w.get(row, col));
				s += "|";
				map.add(s, x, y + row);
			}
			
			x += (1 + l.w.nCol() * (dSize + 1) + 1);
			
			y = (totalHeight - l.b.size()) / 2;
			for (int row = 0; row < l.b.size(); row++)
				map.add(String.format("|%2.2e|", l.b.get(row)), x, y + row);

			x += (1 + dSize + 2);
			
			y = (totalHeight - l.out.size()) / 2;
			for (int row = 0; row < l.out.size(); row++)
				map.add(String.format("|%2.2e|", l.out.get(row)), x, y + row);
			
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