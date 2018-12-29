package com.mlp.program;

import javax.swing.JFrame;

public class ImageFrame extends JFrame {
	private static final long serialVersionUID = 1L;
	private ImagePanel imagePanel;
	
	public ImageFrame() {
		this.imagePanel = new ImagePanel();
		this.getContentPane().add(imagePanel);
		this.setAlwaysOnTop(true);
		this.pack();
	}
	
	public void setImage(double[] data, int width, int height) {
		imagePanel.setImage(data, width, height);
	}
	
	public void setImage(int[][] image) {
		imagePanel.setImage(image);
	}
}