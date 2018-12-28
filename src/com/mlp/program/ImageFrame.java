package com.mlp.program;

import java.awt.Color;
import java.awt.Dimension;
import java.awt.Graphics;
import java.awt.image.BufferedImage;
import javax.swing.JFrame;
import javax.swing.JPanel;

public class ImageFrame extends JFrame {
	private static final long serialVersionUID = 1L;
	private ImagePanel imagePanel;
	
	public ImageFrame() {
		this.imagePanel = new ImagePanel();
		this.getContentPane().add(imagePanel);
		this.setAlwaysOnTop(true);
		this.pack();
	}

	public void setImage(int[][] image) {
		imagePanel.setImage(image);
	}
}

class ImagePanel extends JPanel {
	
	private static final long serialVersionUID = 1L;
	BufferedImage image = new BufferedImage(28, 28, BufferedImage.TYPE_INT_ARGB);
	
	public ImagePanel() {
		this.setPreferredSize(new Dimension(200, 200));
		this.setBackground(Color.BLACK);
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