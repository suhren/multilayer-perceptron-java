package com.mlp.network;

import java.util.ArrayList;
import java.util.List;

import com.mlp.math.Vector;

public class TrainingSet {
	private List<TrainingEntry> trainingEntries = new ArrayList<>();
	
	public TrainingSet() {
		
	}
	
	public void addTrainingEntry(Vector input, Vector expectedOutput) {
		trainingEntries.add(new TrainingEntry(input, expectedOutput));
	}
	public List<TrainingEntry> getTrainingEntries() {
		return trainingEntries;
	}
}