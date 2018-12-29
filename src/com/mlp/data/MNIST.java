package com.mlp.data;

import java.util.ArrayList;
import java.util.List;

import com.mlp.math.Vector;

public class MNIST extends DataSet {
	
	private int[] labels;
	private int[][][] images;
	
	public MNIST(String name, int[] labels, int[][][] images) {
		super(name, labels.length);
		this.labels = labels.clone();
		this.images = images.clone();
	}
	
	public int nImages() {
		return images.length;
	}
	public int nRows() {
		return images[0].length;
	}
	public int nCols() {
		return images[0][0].length;
	}
	
	public int getLabel(int index) {
		return labels[index];
	}
	public int[][] getImage(int index) {
		return images[index];
	}
	public Vector getImageAsVector(int index) {
		return(imageAsVector(images[index]));
	}
	public Vector getLabelAsVector(int index) {
		return(labelAsVector(labels[index]));
	}
	
	private static Vector labelAsVector(int label) {
		double[] result = new double[10];
		result[label] = 1.0;
		return new Vector(result);
	}
	private static Vector imageAsVector(int[][] image) {
		double[] result = new double[image.length * image[0].length];
		for (int row = 0; row < image.length; row++)
			for (int col = 0; col < image[0].length; col++)
				result[row * image[0].length + col] = image[row][col] / 255.0;
		return new Vector(result);
	}

	@Override
	public DataEntry getEntry(int i) {
		return new DataEntry(imageAsVector(images[i]), labelAsVector(labels[i]));
	}

	@Override
	public List<DataEntry> getEntries() {
		List<DataEntry> entries = new ArrayList<>();
		
		for (int i = 0; i < labels.length; i++)
			entries.add(new DataEntry(imageAsVector(images[i]), labelAsVector(labels[i])));
		
		return entries;
	}

	@Override
	public boolean hasImage(int i) {
		return (i >= 0 && i < images.length);
	}
}