package com.mlp.math;

public class AFun {
	@FunctionalInterface
	public interface Function { double eval(double x); }
		
	private final String code;
	private final Function function;
	private final Function derivate;
	
	public AFun(String code, Function function, Function derivate) {
		this.code = code;
		this.function = function;
		this.derivate = derivate;
	}
	
	public String code() { 
		return code;
	}

	public double eval(double x) { 
		return function.eval(x); 
	}
	
	public double evalDerivate(double x) {
		return derivate.eval(x);
	}
	
	public Vector eval(Vector v) { 
		double[] result = new double[v.size()];
		for (int i = 0; i < v.size(); i++)
			result[i] = eval(v.get(i));
		return new Vector(result);
	}
	
	public Vector evalDerivate(Vector v) {
		double[] result = new double[v.size()];
		for (int i = 0; i < v.size(); i++)
			result[i] = evalDerivate(v.get(i));
		return new Vector(result);
	}
}
