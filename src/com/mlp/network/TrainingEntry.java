package com.mlp.network;

public class TrainingEntry {
	private double[] input;
	private double[] expectedOutput;
	
	public TrainingEntry(double[] input, double[] expectedOutput) {
		this.input = input;
		this.expectedOutput = expectedOutput;
	}
	
	public double[] getInput() {
		return input;
	}

	public double[] getExpectedOutput() {
		return expectedOutput;
	}
}