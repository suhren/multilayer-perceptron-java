/**
 * A class representing a Multilayer Perceptron
 * @author Adam
 *
 */
public class MLP {
	private Layer[] layers;
	private Layer outputLayer;
	private double cost;
	
	private double[] input;
	
	public MLP(Layer[] layers) {
		if (layers == null)
			throw new NullPointerException("No layers specified.");
		if (layers.length < 3)
			throw new IllegalArgumentException("The MLP needs to have at least 3 layers.");
		
		// Copy over the layers for the immutable property
		this.layers = layers.clone();
		this.outputLayer = layers[layers.length - 1];
		input = new double[layers[0].w[0].length];
	}
	
	/**
	 * @param The input vector of the MLP.
	 * @param The expected output vector of the MLP.
	 * @return The output of the MLP.
	 */
	public double[] feedThrough(double[] input, double[] expectedOutput) {
		if (outputLayer.o.length != expectedOutput.length)
			throw new IllegalArgumentException("The dimensions of the expected output vector has to be equal to the dimensions of the output vector in the output layer.");
		
		this.input = input;
		double[] current = input.clone();
		
		for (Layer l : layers)
			current = l.feedForward(current);

		cost = calcCost(outputLayer.o, expectedOutput);
		
		return current;
	}
	
	/**
	 * Calculates the Euclidean distance of two vectors.
	 * @param The first vector a.
	 * @param The second vector b.
	 * @return The Eucledian distance of the two vectors.
	 */
	private double calcCost(double[] a, double[] b) {
		double[] d = Layer.vecSub(a, b);
		return Math.sqrt(Layer.dotProduct(d, d));
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
		int dSize = 4;
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
			map.add(String.format("|%.2f|", input[i]), x, y + i);
		
		x += (1 + dSize + 1 + sep + 1);
		
		for (Layer l : layers) {
			y = (totalHeight - l.w.length) / 2;
			for (int row = 0; row < l.w.length; row++) {
				String s = "|";
				s += String.format("%.2f", l.w[row][0]);
				for (int col = 1; col < l.w[0].length; col++)
					s += String.format(" %.2f", l.w[row][col]);
				s += "|";
				map.add(s, x, y + row);
			}
			
			x += (1 + l.w[0].length * (dSize + 1) + 1);
			
			y = (totalHeight - l.b.length) / 2;
			for (int row = 0; row < l.b.length; row++)
				map.add(String.format("|%.2f|", l.b[row]), x, y + row);

			x += (1 + dSize + 2);
			
			y = (totalHeight - l.o.length) / 2;
			for (int row = 0; row < l.o.length; row++)
				map.add(String.format("|%.2f|", l.o[row]), x, y + row);
			
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