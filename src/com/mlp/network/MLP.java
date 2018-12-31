package com.mlp.network;

import com.mlp.data.DataEntry;
import com.mlp.data.DataSet;
import com.mlp.math.AFun;
import com.mlp.math.Vector;

/**
 * A class representing a Multilayer Perceptron.
 * @author Adam
 *
 */
public class MLP {
	private Layer[] layers;
	private Layer oL;
	private double cost;
	private double eta = 0.01;
	private String name;
	
	private Vector input;
	
	public MLP(String name, double eta, int inputSize, int[] outputSizes, AFun aFun) {
		this.name = name;
		layers = new Layer[outputSizes.length];
		int inSize = inputSize;
		for (int i = 0; i < outputSizes.length; i++) {
			layers[i] = new Layer(outputSizes[i], inSize, aFun);
			layers[i].randomize(); 
			inSize = outputSizes[i];
		}
		
		this.oL = layers[layers.length - 1];
		input = new Vector(inputSize);
		this.eta = eta;
	}
	
	public MLP(String name, double eta, Layer[] layers) {
		if (layers == null)
			throw new NullPointerException("No layers specified.");
		if (layers.length < 2)
			throw new IllegalArgumentException("The MLP needs to have at least 2 layers.");
		
		this.name = name;
		this.eta = eta;
		// Copy over the layers for the immutable property
		this.layers = layers.clone();
		this.oL = layers[layers.length - 1];
		input = new Vector(layers[0].w.nCol());
	}
	
	public double train(DataSet trainingSet) {
		double sum = 0.0;
		
		for (DataEntry e : trainingSet.getEntries()) {
			feedThrough(e.getInput(), e.getExpected());
			sum += cost;
			backPropagate(e.getExpected());
		}
		
		return sum / trainingSet.getEntries().size();
	}
	
	public double train(DataSet trainingSet, int n) {
		double sum = 0.0;
		
		for (int i = 0; i < n; i++)
			sum += train(trainingSet);
		
		return sum / n;
	}
	
	public int test(DataSet testSet) {
		int nCorrect = 0;
		
		for (DataEntry e : testSet.getEntries()) {
			feedThrough(e.getInput(), e.getExpected());
			if (mostLikely(oL.out) == mostLikely(e.getExpected()))
				nCorrect++;
		}
		
		return nCorrect;
		
	}
	
	/**
	 * @param The input vector of the MLP.
	 * @param The expected output vector of the MLP.
	 * @return The output of the MLP.
	 */
	public Vector feedThrough(Vector input, Vector expectedOutput) {
		if (oL.out.size() != expectedOutput.size())
			throw new IllegalArgumentException("The dimensions of the expected output vector has to be equal to the dimensions of the output vector in the output layer.");
		
		this.input = input.clone();
		Vector current = input.clone();
		
		for (Layer l : layers)
			current = l.feedForward(current);

		cost = calcCost(expectedOutput, oL.out);
		
		return current;
	}
	
	/**
	 * Perform back propagation updating the weights of the layers using the latest output of MLP.
	 */
	public void backPropagate(Vector eO) {
		// Output layer
		for (int row = 0; row < oL.w.nRow(); row++) {
			double dEdO = 2 * (oL.out.get(row) - eO.get(row));
			double dOdZ = oL.aFun.evalDerivate(oL.z.get(row));
			oL.dEdZ.set(row, dEdO * dOdZ);
			double dEdB = dEdO * dOdZ * 1;
			oL.bNew.set(row, oL.b.get(row) - eta * dEdB);
			for (int col = 0; col < oL.w.nCol(); col++) {
				double dZdW = oL.in.get(col);
				double dEdW = dEdO * dOdZ * dZdW;
				oL.wNew.set(row, col, oL.w.get(row, col) - eta * dEdW);
			}
		}
		
		// Hidden layers
		for (int i = layers.length - 2; i >= 0; i--) {
			Layer l = layers[i];
			Layer nl = layers[i + 1];
			for (int row = 0; row < l.w.nRow(); row++) {
				double dOdZ = l.aFun.evalDerivate(l.z.get(row));
				double dEdO = 0;
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
	
	public Vector getInput() {
		return input;
	}
	
	public Vector getOutput() {
		return oL.out;
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
	
	public String getName() {
		return (name != null) ? name : "N/A";
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

	public Layer getLayer(int i) {
		return layers[i];
	}
		
	public int mostLikely(Vector output) {
		int current = 0;
		for (int i = 1; i < output.size(); i++)
			if (output.get(i) > output.get(current))
				current = i;
		return current;
	}
	
	public double getLearningRate() {
		return eta;
	}

	public void setName(String name) {
		this.name = name;
	}
}