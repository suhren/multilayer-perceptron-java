/**
 * A class representing a Multilayer Perceptron
 * @author Adam
 *
 */
public class MLP {
	private Layer[] layers;
	private double[] input;
	
	public MLP(Layer[] layers) {
		if (layers == null)
			throw new NullPointerException("No layers specified.");
		if (layers.length < 3)
			throw new IllegalArgumentException("The MLP needs to have at least 3 layers.");
		
		// Copy over the layers for immutability
		this.layers = layers.clone();
		input = new double[layers[0].w[0].length];
	}
	
	public double[] feedThrough(double[] data) {
		input = data;
		double[] current = data.clone();
		
		for (Layer l : layers)
			current = l.feedForward(current);
		
		return current;
	}
	
	@Override
	public String toString() {
		StringBuilder builder = new StringBuilder();
		for (Double d : input)
			builder.append(d + "\n");
		for (Layer l : layers)
			builder.append(l.toString());
		return builder.toString();
	}
}