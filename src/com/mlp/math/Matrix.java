package com.mlp.math;

public class Matrix {
	private double[][] data;
	private final int nRow;
	private final int nCol;
	
	public Matrix(int nRows, int nCols) {
		this(new double[nRows][nCols]);
	}
	
	public Matrix(double[][] data) {
		if (data == null || data[0] == null)
			throw new NullPointerException("Data not specified.");
		this.data = data.clone();
		this.nRow = data.length;
		this.nCol = data[0].length;
	}
	
	public Matrix clone() {
		return new Matrix(data.clone());
	}
	
	public Vector multiply(Vector v) {
		if (nCol != v.size())
			throw new IllegalArgumentException("Columns in matrix != elements in vector.");
		
		double[] result = new double[nRow];
		
		for (int row = 0; row < nRow; row++)
			result[row] = v.dotProduct(data[row]);
		
		return new Vector(result);
	}
	
	public void set(int row, int col, double val) {
		data[row][col] = val;
	}

	public void setRow(int row, double[] data) {
		this.data[row] = data.clone();
	}
	public double get(int row, int col) {
		return data[row][col];
	}
	
	public int nRow() {
		return nRow;
	}
	
	public int nCol() {
		return nCol;
	}
}
