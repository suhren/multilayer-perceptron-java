package com.mlp.network;

import java.util.ArrayList;
import java.util.List;

import com.mlp.math.Vector;

public class DataSet {
	private List<DataEntry> entries = new ArrayList<>();
	
	public void addEntry(Vector input, Vector expected) {
		entries.add(new DataEntry(input, expected));
	}
	public List<DataEntry> getEntries() {
		return entries;
	}
}