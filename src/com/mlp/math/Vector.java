package com.mlp.math;

public class Vector {
	private double[] data;
	private final int size;
	
	public Vector(int size) {
		this(new double[size]);
	}
	public Vector(double[] data) {
		if (data == null)
			throw new NullPointerException("Data not specified.");
		this.data = data.clone();
		this.size = data.length;
	}
	
	public Vector clone() {
		return new Vector(data.clone());
	}
	
	public void set(int i, double val) {
		data[i] = val;
	}

	public double get(int i) {
		return data[i];
	}
	
	public Vector add(Vector v) {
		if (size != v.size())
			throw new NullPointerException("Vector is null.");
		if (size != v.size())
			throw new IllegalArgumentException("Vectors have to be of the same length.");
		
		double[] result = new double[size];
		
		for (int i = 0; i < size; i++)
			result[i] = data[i] + v.data[i];
		
		return new Vector(result);
	}
	
	public Vector sub(Vector v) {
		if (v == null)
			throw new NullPointerException("Vector is null.");
		if (size != v.size())
			throw new IllegalArgumentException("Vectors have to be of the same length.");
		
		double[] result = new double[size];
		
		for (int i = 0; i < size; i++)
			result[i] = data[i] - v.data[i];
		
		return new Vector(result);
	}
	
	public Vector multiply(double d) {
		double[] result = new double[size];
		
		for (int i = 0; i < size; i++)
			result[i] *= d;
		
		return new Vector(result);
	}
	
	public Vector multiply(Vector v) {
		if (v == null)
			throw new NullPointerException("Vector is null.");
		if (size != v.size())
			throw new IllegalArgumentException("Vectors have to be of the same length.");
		
		double[] result = new double[size];
		
		for (int i = 0; i < size; i++)
			result[i] = data[i] * v.get(i);
		
		return new Vector(result);
	}
	
	public double dotProduct(Vector v) {
		return dotProduct(v.data);
	}
	
	public double dotProduct(double[] x) {
		if (data.length != x.length)
			throw new IllegalArgumentException("Vectors have to be of the same length.");
		
		double result = 0.0;
		for (int i = 0; i < data.length; i++)
			result += data[i] * x[i];
		return result;
	}
	
	public int size() {
		return size;
	}
}
