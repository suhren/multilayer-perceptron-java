package com.mlp.data;

import com.mlp.math.Vector;

public class DataEntry {
	private Vector input;
	private Vector expected;
	
	public DataEntry(Vector input, Vector expected) {
		this.input = input;
		this.expected = expected;
	}
	
	public Vector getInput() {
		return input;
	}

	public Vector getExpected() {
		return expected;
	}
}