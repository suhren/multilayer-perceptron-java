package com.mlp.program;

import com.mlp.network.TrainingSet;

public class MNIST extends TrainingSet {

	private int[] testLabels;
	private int[] trainingLabels;
	private int[][][] testImages;
	private int[][][] trainingImages;
	
	public MNIST(int[] trainingLabels, int[][][] trainingImages) {
		this(trainingLabels, trainingImages, null, null);
	}
	public MNIST(int[] trainingLabels, int[][][] trainingImages, int[] testLabels, int[][][] testImages) {
		this.trainingLabels = trainingLabels.clone();
		this.trainingImages = trainingImages.clone();
		this.testLabels = testLabels.clone();
		this.testImages = testImages.clone();
	}
	
	public int nTrainingImages() {
		return trainingImages.length;
	}
	public int nTestImages() {
		return testImages.length;
	}
	public int nRows() {
		return trainingImages[0].length;
	}
	public int nCols() {
		return trainingImages[0][0].length;
	}
	
	public int getTrainingLabel(int index) {
		return trainingLabels[index];
	}
	public int[][] getTrainingImage(int index) {
		return trainingImages[index];
	}
	public double[] getTrainingImageAsVector(int index) {
		return(imageAsVector(trainingImages[index]));
	}
	public double[] getTrainingLabelAsVector(int index) {
		return(labelAsVector(trainingLabels[index]));
	}
	
	public int getTestLabel(int index) {
		return testLabels[index];
	}
	public int[][] getTestImage(int index) {
		return testImages[index];
	}
	public double[] getTestImageAsVector(int index) {
		return(imageAsVector(testImages[index]));
	}
	public double[] getTestLabelAsVector(int index) {
		return(labelAsVector(testLabels[index]));
	}
	
	private static double[] labelAsVector(int label) {
		double[] result = new double[10];
		result[label] = 1.0;
		return result;
	}
	private static double[] imageAsVector(int[][] image) {
		double[] result = new double[image.length * image[0].length];
		for (int row = 0; row < image.length; row++)
			for (int col = 0; col < image[0].length; col++)
				result[row * image[0].length + col] = image[row][col] / 255.0;
		return result;
	}
}