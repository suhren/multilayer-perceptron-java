package com.mlp.data;

import java.util.List;

public abstract class DataSet {
	private String name;
	private int size;
	
	public DataSet(String name, int size) {
		this.name = name;
		this.size = size;
	}
	
	public String getName() {
		return name;
	}
	
	public int getSize() {
		return size;
	}

	public abstract boolean hasImage(int i);
	public abstract int[][] getImage(int i);
	public abstract DataEntry getEntry(int i);
	public abstract List<DataEntry> getEntries();
}