package com.mlp.program;

import java.awt.Dimension;
import java.awt.FlowLayout;
import java.awt.event.ActionEvent;
import java.text.NumberFormat;

import javax.swing.AbstractAction;
import javax.swing.Action;
import javax.swing.BorderFactory;
import javax.swing.BoxLayout;
import javax.swing.InputMap;
import javax.swing.JButton;
import javax.swing.JComboBox;
import javax.swing.JComponent;
import javax.swing.JDialog;
import javax.swing.JFormattedTextField;
import javax.swing.JFrame;
import javax.swing.JOptionPane;
import javax.swing.JPanel;
import javax.swing.JRootPane;
import javax.swing.JTextField;
import javax.swing.KeyStroke;

import com.mlp.math.Utils;
import com.mlp.network.Layer;
import com.mlp.network.MLP;

class ProtoLayer {
	int number;
	int size;
	String aFun;
	
	public ProtoLayer(int index, int size, String aFun) {
		this.number = index;
		this.size = size;
		this.aFun = aFun;
	}
	
	public String toString() {
		return "Layer " + number;
	}
}

public class NewMLPDialog extends JDialog {
	private static final long serialVersionUID = 1L;
	private JFormattedTextField tfRate;
	private JFormattedTextField tfInput;
	private JComboBox<ProtoLayer> comboLayers;
	private JFormattedTextField tfLayerOutput;
	private JComboBox<String> comboAFun;
	private MLP mlp;
	
	public NewMLPDialog(JFrame owner) {
		super(owner, true);
		setResizable(false);
		setTitle("New MLP Dialog");
        setDefaultCloseOperation(DISPOSE_ON_CLOSE);
		
		getContentPane().setLayout(new BoxLayout(getContentPane(), BoxLayout.Y_AXIS));
		
		tfRate = new JFormattedTextField(NumberFormat.getNumberInstance());
		tfRate.setValue(0.1);
		tfRate.setBorder(BorderFactory.createTitledBorder("Learning rate"));
		getContentPane().add(tfRate);
		
		tfInput = new JFormattedTextField(NumberFormat.getNumberInstance());
		tfInput.setValue(10);
		tfInput.setBorder(BorderFactory.createTitledBorder("Input size"));
		getContentPane().add(tfInput);
		
		JPanel layerPanel = new JPanel();
		layerPanel.setBorder(BorderFactory.createTitledBorder("Layers"));
		layerPanel.setLayout(new BoxLayout(layerPanel, BoxLayout.Y_AXIS));
		getContentPane().add(layerPanel);
		
		JPanel comboPanel = new JPanel();
		comboPanel.setLayout(new FlowLayout());
		layerPanel.add(comboPanel);
		
		JButton newLayer = new JButton("New");
		newLayer.addActionListener(e -> {
			ProtoLayer p = new ProtoLayer(comboLayers.getItemCount() + 1, 8, "");
			comboLayers.addItem(p);
			comboLayers.setSelectedItem(p);
			comboAFun.setSelectedIndex(0);
			if (comboLayers.getItemCount() > 0) {
				tfLayerOutput.setEnabled(true);
				comboAFun.setEnabled(true);
			}
		});
		comboPanel.add(newLayer);
		
		JButton deleteLayer = new JButton("Delete");
		deleteLayer.addActionListener(e -> {
			ProtoLayer selected = (ProtoLayer) comboLayers.getSelectedItem();
			for (int i = 0; i < comboLayers.getItemCount(); i++) {
				if (comboLayers.getItemAt(i).number > selected.number)
					comboLayers.getItemAt(i).number--;
			}
					
			comboLayers.removeItem(selected);
			
			if (comboLayers.getItemCount() == 0) {
				tfLayerOutput.setEnabled(false);
				comboAFun.setEnabled(false);
			}
		});
		comboPanel.add(deleteLayer);
		
		comboLayers = new JComboBox<>();
		comboLayers.addActionListener(e -> {
			if (comboLayers.getSelectedItem() != null) {
				ProtoLayer p = (ProtoLayer) comboLayers.getSelectedItem();
				tfLayerOutput.setValue(p.size);
				comboAFun.setSelectedItem(p.aFun);
			}
		});
		Dimension d = comboLayers.getPreferredSize();
		d.width = 100;
		comboLayers.setPreferredSize(d);
		comboPanel.add(comboLayers);
		
		tfLayerOutput = new JFormattedTextField(NumberFormat.getNumberInstance());
		tfLayerOutput.setEnabled(false);
		tfLayerOutput.addActionListener(e -> {
			ProtoLayer p = (ProtoLayer) comboLayers.getSelectedItem();
			p.size = (int) (long) tfLayerOutput.getValue();
		});
		tfLayerOutput.setBorder(BorderFactory.createTitledBorder("Output size"));
		layerPanel.add(tfLayerOutput);
		
		comboAFun = new JComboBox<>(Utils.AFunLibrary.getCodes());
		comboAFun.setEnabled(false);
		comboAFun.addActionListener(e -> {
			ProtoLayer p = (ProtoLayer) comboLayers.getSelectedItem();
			p.aFun = (String) comboAFun.getSelectedItem();
		});
		comboAFun.setBorder(BorderFactory.createTitledBorder("Activation function"));
		layerPanel.add(comboAFun);
		
		JPanel buttonPanel = new JPanel();
		buttonPanel.setLayout(new FlowLayout());
		getContentPane().add(buttonPanel);
		
		JButton okButton = new JButton("Ok");
		okButton.addActionListener(e -> {
			createMLP();
			closeDialog();
		});
		buttonPanel.add(okButton);
		
		JButton cancelButton = new JButton("Cancel");
		cancelButton.addActionListener(e -> closeDialog());
		buttonPanel.add(cancelButton);
				
		pack();
	}
	
	private void closeDialog() {
		setVisible(false);
		dispose();
	}
	
	private void createMLP() {
		try {
			int prevSize = Integer.parseInt(tfInput.getText());
			Layer[] layers = new Layer[comboLayers.getItemCount()];
			for (int i = 0; i < comboLayers.getItemCount(); i++) {
				ProtoLayer p = (ProtoLayer) comboLayers.getItemAt(i);
				layers[i] = new Layer(p.size, prevSize, Utils.AFunLibrary.fromCode(p.aFun));
				prevSize = p.size;
				layers[i].randomize();
			}
			mlp = new MLP(null, (double) tfRate.getValue(), layers);
		}
		catch(Exception e) {
			JOptionPane.showMessageDialog(this, "Failed to create MLP", "Error", JOptionPane.ERROR_MESSAGE);
		}
	}

	/**
	 * Override the createRootPane inherited by the JDialog, to create the rootPane.
	 * Create functionality to close the window when "Escape" button is pressed.
	 * https://examples.javacodegeeks.com/desktop-java/swing/jdialog/java-jdialog-example/
	 */
	@Override
	public JRootPane createRootPane() {
		JRootPane rootPane = new JRootPane();
		KeyStroke stroke = KeyStroke.getKeyStroke("ESCAPE");
		Action action = new AbstractAction() {
			private static final long serialVersionUID = 1L;
			public void actionPerformed(ActionEvent e) {
				closeDialog();
			}
		};
		InputMap inputMap = rootPane.getInputMap(JComponent.WHEN_IN_FOCUSED_WINDOW);
		inputMap.put(stroke, "ESCAPE");
		rootPane.getActionMap().put("ESCAPE", action);
		return rootPane;
	}

	public MLP getMLP() {
		return mlp;
	}
}
