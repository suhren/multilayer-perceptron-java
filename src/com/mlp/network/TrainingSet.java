package com.mlp.network;

import java.util.ArrayList;
import java.util.List;

public class TrainingSet {
	private List<TrainingEntry> trainingEntries = new ArrayList<>();
	
	public TrainingSet() {
		
	}
	
	public void addTrainingEntry(double[] input, double[] expectedOutput) {
		trainingEntries.add(new TrainingEntry(input, expectedOutput));
	}
	public List<TrainingEntry> getTrainingEntries() {
		return trainingEntries;
	}
}