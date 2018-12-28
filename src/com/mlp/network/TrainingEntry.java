package com.mlp.network;

import com.mlp.math.Vector;

public class TrainingEntry {
	private Vector input;
	private Vector expectedOutput;
	
	public TrainingEntry(Vector input, Vector expectedOutput) {
		this.input = input;
		this.expectedOutput = expectedOutput;
	}
	
	public Vector getInput() {
		return input;
	}

	public Vector getExpectedOutput() {
		return expectedOutput;
	}
}