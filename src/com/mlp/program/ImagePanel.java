package com.mlp.program;

import java.awt.Color;
import java.awt.Dimension;
import java.awt.Graphics;
import java.awt.image.BufferedImage;

import javax.swing.JPanel;

import com.mlp.math.Utils;

public class ImagePanel extends JPanel {
	
	private static final long serialVersionUID = 1L;
	BufferedImage image = new BufferedImage(28, 28, BufferedImage.TYPE_INT_ARGB);
	
	public ImagePanel() {
		this.setPreferredSize(new Dimension(300, 300));
		this.setBackground(Color.BLACK);
	}
	
	public void setImage(double[] data, int width, int height) {
		double max = Utils.max(data);
		image = new BufferedImage(width, height, BufferedImage.TYPE_INT_ARGB);
		for (int i = 0; i < data.length; i++) {
			float w = (float) Utils.constrain(data[i] / max, 0.0, 1.0);
			Color c = new Color(w, w, w);
			image.setRGB(i % width, i / width, c.getRGB());
		}
		this.paintImmediately(getBounds());
	}
	
	public void setImage(int[][] data) {
		image = new BufferedImage(data[0].length, data.length, BufferedImage.TYPE_INT_ARGB);
		for (int row = 0; row < data.length; row++)
			for (int col = 0; col < data[0].length; col++) {
				Color c = new Color(data[row][col], data[row][col], data[row][col], 255);
				image.setRGB(col, row, c.getRGB());
			}
		this.paintImmediately(getBounds());
	}
	
	@Override
	public void paintComponent(Graphics g) {
		super.paintComponents(g);
        g.drawImage(image, 0, 0, this.getWidth(), this.getHeight(), null);
	}
}