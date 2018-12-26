package com.mlp.math;

public class ActivationFunction {
	@FunctionalInterface
	public interface Function { double eval(double x); }
	
	private Function function;
	private Function derivate;
	
	public ActivationFunction(Function function, Function derivate) {
		this.function = function;
		this.derivate = derivate;
	}
	
	public double eval(double x) {
		return function.eval(x);
	}
	
	public double evalDerivate(double x) {
		return derivate.eval(x);
	}
}
