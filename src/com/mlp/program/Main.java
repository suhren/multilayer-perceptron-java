package com.mlp.program;
import java.util.Scanner;

import com.mlp.math.Vector;
import com.mlp.math.Utils;
import com.mlp.network.MLP;
import com.mlp.network.DataSet;

/**
 * The main entry point of the program.
 * @author Adam
 */
public class Main {
	public static void main(String[] args) {
		MNIST mnist = FileUtils.readMNIST(
				"MNIST\\train-labels-idx1-ubyte",
				"MNIST\\train-images-idx3-ubyte",
				"MNIST\\t10k-labels-idx1-ubyte",
				"MNIST\\t10k-images-idx3-ubyte");
		
		DataSet trainingSet = new DataSet();
		for (int i = 0; i < mnist.nTrainingImages(); i++)
			trainingSet.addEntry(mnist.getTrainingImageAsVector(i), mnist.getTrainingLabelAsVector(i));

		DataSet testSet = new DataSet();
		for (int i = 0; i < mnist.nTestImages(); i++)
			testSet.addEntry(mnist.getTestImageAsVector(i), mnist.getTestLabelAsVector(i));
		
		// Works for a single image, not all of them!
		//MLP mlp = new MLP(mnist.nRows() * mnist.nCols(), new int[] {8, 4, 4, 10}, Utils.sinusoid, 0.01);
		
//		MLP mlp = new MLP("mlp-12-10-elliotSig-0_01", 0.01, mnist.nRows() * mnist.nCols(), new int[] {12, 10}, Utils.AFunLibrary.ELLIOT_SIG.aFun());

		MLP mlp = FileUtils.readFromFile("mlp-12-10-elliotSig-0_01.txt");
		
//		for (int i = 1; i <= 100; i++) {
//			double averageCost = mlp.train(trainingSet);
//			System.out.println("Loop" + i + ": Cost: " + averageCost);
//		}
	
		double rate = mlp.test(testSet);
		System.out.println("The MLP guessed the correct digit in " + rate * 100 + "% of the cases.");
		rate = mlp.test(trainingSet);
		System.out.println("The MLP guessed the correct digit in " + rate * 100 + "% of the cases.");
		
		ImageFrame imageFrame = new ImageFrame();
		imageFrame.setVisible(true);

//		FileUtils.writeToFile(mlp, "mlp-12-10-elliotSig-0_01.txt");
		
		int i = -1;
		Scanner sc = new Scanner(System.in);
		
		while (true) {
			String input = sc.nextLine();
			String[] parts = input.split(" ");
			if (parts[0].equals("load")) {
				i = Integer.parseInt(parts[1]);
				imageFrame.setImage(mnist.getTrainingImage(i));
				imageFrame.setTitle(Integer.toString(mnist.getTrainingLabel(i)));
			}
			else if (parts[0].equals("input")) {
				if (i > 0) {
					Vector output = mlp.feedThrough(mnist.getTrainingImageAsVector(i), mnist.getTrainingLabelAsVector(i));
					printVector(output);
					System.out.println("The most likely digit is " + mostLikely(output));
				}
				else {
					System.out.println("Invalid index");
				}
			}
			else if (parts[0].equals("exit")) {
				break;
			}
		}
		sc.close();
		
//		
//		System.out.println("");
//		printArray(mnist.getTrainingImageAsVector(0));
//		
//		ImageFrame imageFrame = new ImageFrame();
//		imageFrame.setVisible(true);
//		for (int i = 0; i < 100; i++) {
//			imageFrame.setImage(mnist.getTrainingImage(i));
//			imageFrame.setTitle(Integer.toString(mnist.getTrainingLabel(i)));
//			try {
//				Thread.currentThread().sleep(1000);
//			} catch (InterruptedException e) {
//				// TODO Auto-generated catch block
//				e.printStackTrace();
//			}
//		}
		
//		Vector input = new Vector(new double[] {1.0, 2.0, 3.0, 4.0});
//		Vector expected = new Vector(new double[] {1.0, 0.0, 1.0, 0.0, -1.0});
//		String filePath = "TestMLP.txt";
//		
//		MLP network = FileUtils.readFromFile(filePath);
//		System.out.println("Initial state:");
//		System.out.println(network.getCost());
//		System.out.print(network.toString() + "\n");
//		System.out.println("Feedthrough:");
//		network.feedThrough(input, expected);
//		System.out.println(network.getCost());
//		System.out.print(network.toString() + "\n");
//		for (int i = 1; i <= 1000; i++) {
//			System.out.println("Backpropogation " + i + ": ");
//			network.backPropagate(expected);
//			System.out.print(network.toString() + "\n");
//			System.out.println("New Feedthrough:");
//			network.feedThrough(input, expected);
//			System.out.println(network.getCost());
//			System.out.print(network.toString());
//		}
	}
	
	public static int mostLikely(Vector output) {
		int current = 0;
		for (int i = 1; i < output.size(); i++)
			if (output.get(i) > output.get(current))
				current = i;
		return current;
	}
	
	public static void printVector(Vector d) {
		for (int i = 0; i < d.size(); i++)
			System.out.println(d.get(i));
	}
}