package com.mlp.math;

import java.util.Arrays;

public class Utils {
	// List of activation functions:
	// https://en.wikipedia.org/wiki/Activation_function
	
	public enum AFunLibrary {
		IDENTITY(new AFun(
			"Identity",
			(x) -> x,
			(x) -> 1)),
		BINARY_STEP(new AFun(
			"Binary Step",
			(x) -> (x < 0) ? 0 : 1,
			(x) -> 0)),
		LOGISTIC(new AFun(
			"Logistic",
			(x) -> 1.0/(1 + Math.exp(-x)),
			(x) -> 1.0/(1 + Math.exp(-x)) * (1 - 1.0/(1 + Math.exp(-x))))),
		TANH(new AFun(
			"TanH",
			(x) -> Math.tanh(x),
			(x) -> 1 - Math.pow(Math.tanh(x), 2))),
		ARCTAN(new AFun(
			"ArcTan",
			(x) -> Math.atan(x),
			(x) -> 1.0 / (x*x + 1))),
		ELLIOT_SIG(new AFun(
			"ElliotSig",
			(x) -> x / (1 + Math.abs(x)),
			(x) -> 1.0 / Math.pow(1 + Math.abs(x), 2))),
		RELU(new AFun(
			"ReLU",
			(x) -> (x < 0) ? 0 : x,
			(x) -> (x < 0) ? 0 : 1)),
		RELU_LEAKY(new AFun(
			"ReLULeaky",
			(x) -> (x < 0) ? 0.01 * x : x,
			(x) -> (x < 0) ? 0.01 : 1)),
		SOFT_PLUS(new AFun(
			"SoftPlus",
			(x) -> Math.log(1 + Math.exp(x)),
			(x) -> 1.0 / (1 + Math.exp(-x)))),
		SINUSOID(new AFun(
			"Sinusoid",
			(x) -> Math.sin(x),
			(x) -> Math.cos(x))),
		SINC(new AFun(
			"Sinc",
			(x) -> (x == 0) ? 0 : Math.sin(x) / x,
			(x) -> (x == 0) ? 0 : Math.cos(x) / x - Math.sin(x) / x*x)),
		GAUSSIAN(new AFun(
			"Gaussian",
			(x) -> Math.exp(-x*x),
			(x) -> -2*x*Math.exp(-x*x)));
		
		private final AFun aFun;
		
		AFunLibrary(AFun aFun) {
			this.aFun = aFun;
		}
		
		public static AFun fromCode(String code) {
			if (code != null)
				for (AFunLibrary a : AFunLibrary.values())
					if (code.equalsIgnoreCase(a.aFun.code()))
						return a.aFun;
			return null;
		}
		
		public String code() {
			return aFun.code();
		}
		
		public AFun aFun() {
			return aFun;
		}

		public static String[] getCodes() {
			return Arrays.stream(AFunLibrary.values()).map(AFunLibrary::code).toArray(String[]::new);
		}
	}
	
	public static double max(double[] data) {
		double max = data[0];
		for (int i = 0; i < data.length; i++)
			if (data[i] > max)
				max = data[i];
		return max;
	}
	
	public static int constrain(int val, int min, int max) {
		if (val < min)
			return min;
		if (val > max)
			return max;
		return val;
	}
	
	public static double constrain(double val, double min, double max) {
		if (val < min)
			return min;
		if (val > max)
			return max;
		return val;
	}
}