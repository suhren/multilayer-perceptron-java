package com.mlp.program;
import com.mlp.math.Utils;
import com.mlp.network.MLP;

/**
 * The main entry point of the program.
 * @author Adam
 */
public class Main {
	public static void main(String[] args) {
		double[] input = new double[] {1.0, 2.0, 3.0, 4.0};
		double[] expected = new double[] {1.0, 0.0, 1.0, 0.0, -1.0};
		String filePath = "TestMLP2.txt";
		MLP network = new MLP(FileUtils.readFromFile(filePath, Utils.sinusoid));
		System.out.println("Initial state:");
		System.out.println(network.getCost());
		System.out.print(network.toString() + "\n");
		System.out.println("Feedthrough:");
		network.feedThrough(input, expected);
		System.out.println(network.getCost());
		System.out.print(network.toString() + "\n");
		for (int i = 1; i <= 10000; i++) {
			System.out.println("Backpropogation " + i + ": ");
			network.backPropagate(expected);
			System.out.print(network.toString() + "\n");
			System.out.println("New Feedthrough:");
			network.feedThrough(input, expected);
			System.out.println(network.getCost());
			System.out.print(network.toString());
		}
//		FileUtils.writeToFile(network, "TestMLP2.txt");
	}
}