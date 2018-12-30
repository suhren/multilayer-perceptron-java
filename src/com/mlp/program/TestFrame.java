package com.mlp.program;

import java.awt.BorderLayout;
import java.awt.Color;
import java.awt.FlowLayout;
import java.io.File;
import java.io.FileFilter;
import java.text.NumberFormat;
import java.util.HashMap;
import java.util.Map;

import javax.swing.BorderFactory;
import javax.swing.BoxLayout;
import javax.swing.JButton;
import javax.swing.JComboBox;
import javax.swing.JFileChooser;
import javax.swing.JFormattedTextField;
import javax.swing.JFrame;
import javax.swing.JLabel;
import javax.swing.JOptionPane;
import javax.swing.JPanel;
import javax.swing.JScrollBar;
import javax.swing.JScrollPane;
import javax.swing.JTextArea;
import javax.swing.SwingUtilities;
import javax.swing.filechooser.FileNameExtensionFilter;
import javax.swing.text.DefaultCaret;

import com.mlp.data.DataEntry;
import com.mlp.data.DataSet;
import com.mlp.data.MNIST;
import com.mlp.math.Utils;
import com.mlp.network.Layer;
import com.mlp.network.MLP;

public class TestFrame extends JFrame {
	private static final long serialVersionUID = 1L;

	private MNIST mnistTraining = FileUtils.readMNIST(
			"MNIST_TRAINING",
			"MNIST\\train-labels-idx1-ubyte",
			"MNIST\\train-images-idx3-ubyte");
	private MNIST mnistTest = FileUtils.readMNIST(
			"MNIST_TEST",
			"MNIST\\t10k-labels-idx1-ubyte",
			"MNIST\\t10k-images-idx3-ubyte");
	
	private Map<String, DataSet> dataSets = new HashMap<>();
	
	private MLP mlp;
	private int dataSetIndex = 0;
	private DataSet dataSet;
	private JFormattedTextField dataIndexField;
	private JLabel sizeLabel;
	private ImagePanel imagePanel;
	private JTextArea dataArea, infoArea, outputArea;
	private JScrollPane scrollAreaLog;
	private JFormattedTextField repeatField;
	
	public TestFrame() {
		setupFrame();
	}
	
	private void setupFrame() {
		setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
		
		dataSets.put(mnistTraining.getName(), mnistTraining);
		dataSets.put(mnistTest.getName(), mnistTest);
		
		JPanel topPanel = new JPanel();
		topPanel.setBorder(BorderFactory.createTitledBorder("Actions"));
		topPanel.setLayout(new FlowLayout());
		JButton newButton = new JButton("New MLP");
		newButton.addActionListener(e -> newMLP());
		topPanel.add(newButton);
		JButton loadButton = new JButton("Load MLP");
		loadButton.addActionListener(e -> loadMLP());
		topPanel.add(loadButton);
		JButton saveButton = new JButton("Save MLP");
		saveButton.addActionListener(e -> saveMLP());
		topPanel.add(saveButton);
		JButton inputEntry = new JButton("Input Entry");
		inputEntry.addActionListener(e -> inputEntry());
		topPanel.add(inputEntry);
		JButton testDataSet = new JButton("Test Data Set");
		testDataSet.addActionListener(e -> inputDataSet());
		topPanel.add(testDataSet);
		JButton trainDataSet = new JButton("Train Data Set");
		trainDataSet.addActionListener(e -> trainDataSet());
		topPanel.add(trainDataSet);
		repeatField = new JFormattedTextField(NumberFormat.getNumberInstance());
		repeatField.setColumns(6);
		repeatField.setBorder(BorderFactory.createTitledBorder("Repeat"));
		repeatField.setValue(1);
		topPanel.add(repeatField);
		JButton clearLog = new JButton("Clear Log");
		clearLog.addActionListener(e -> outputArea.setText(""));
		topPanel.add(clearLog);
		
		getContentPane().add(topPanel, BorderLayout.NORTH);

		JPanel inputPanel = new JPanel();
		inputPanel.setLayout(new BoxLayout(inputPanel, BoxLayout.Y_AXIS));
		inputPanel.setBorder(BorderFactory.createTitledBorder("Input"));
		getContentPane().add(inputPanel, BorderLayout.WEST);

		JComboBox<String> comboBox = new JComboBox<>(dataSets.keySet().toArray(new String[dataSets.size()]));
		comboBox.addActionListener(e -> selectData((String) comboBox.getSelectedItem()));
		inputPanel.add(comboBox);
		
		dataArea = new JTextArea(10, 20);
		dataArea.setEditable(false);
		dataArea.setLineWrap(true);
		dataArea.setBackground(new Color(240, 240, 240));
		dataArea.setBorder(BorderFactory.createTitledBorder("Data info"));
		inputPanel.add(dataArea);
		
		JPanel inputControlPanel = new JPanel();
		inputPanel.add(inputControlPanel);
		inputControlPanel.setLayout(new FlowLayout());
		JButton prev = new JButton("Previous");
		prev.addActionListener(e -> prevData());
		inputControlPanel.add(prev);
		dataIndexField = new JFormattedTextField(NumberFormat.getNumberInstance());
		dataIndexField.setColumns(6);
		dataIndexField.setValue(0);
		dataIndexField.addActionListener(e -> indexEdit());
		inputControlPanel.add(dataIndexField);
		sizeLabel = new JLabel();
		inputControlPanel.add(sizeLabel);
		JButton next = new JButton("Next");
		next.addActionListener(e -> nextData());
		inputControlPanel.add(next);
		
		JPanel outerImagePanel = new JPanel();
		outerImagePanel.setBorder(BorderFactory.createTitledBorder("Input image"));
		imagePanel = new ImagePanel();
		outerImagePanel.add(imagePanel);
		inputPanel.add(outerImagePanel);
		
		infoArea = new JTextArea(20, 20);
		infoArea.setBorder(BorderFactory.createTitledBorder("MLP info"));
		getContentPane().add(infoArea, BorderLayout.CENTER);
		
		outputArea = new JTextArea(20, 40);
		outputArea.setEditable(false);
		outputArea.setLineWrap(true);
		outputArea.setBackground(new Color(240, 240, 240));
		outputArea.setBorder(BorderFactory.createTitledBorder("MLP output"));
		// Causes AWT to freeze when called from another thread?
		// DefaultCaret caret = (DefaultCaret) outputArea.getCaret();
		//caret.setUpdatePolicy(DefaultCaret.ALWAYS_UPDATE);
		scrollAreaLog = new JScrollPane(outputArea, JScrollPane.VERTICAL_SCROLLBAR_ALWAYS,
				JScrollPane.HORIZONTAL_SCROLLBAR_AS_NEEDED);
		getContentPane().add(scrollAreaLog, BorderLayout.EAST);

		selectData((String) comboBox.getSelectedItem());
		updateIndexField();
		pack();
	}

	private void trainDataSet() {
		if (mlp == null) {
			printLineLog("No MLP specified");
			return;
		}
		if (dataSet == null) {
			printLineLog("No data set specified");
			return;
		}
		
		int nRepeat = Integer.parseInt(repeatField.getText());
		
		new Thread() {
			@Override
	        public void run() {
				printStatus(mlp, "Started training with " + dataSet.getName() + " repeated " + nRepeat + " times");
				for (int i = 1; i <= nRepeat; i++) {
					double average = mlp.train(dataSet);
					printStatus(mlp, "Set " + i + "/" + nRepeat  + ", ave.cost = " + average);
				}
				printStatus(mlp, "Finished training with " + dataSet.getName());
	        }
		}.start();
	}

	private void saveMLP() {
		if (mlp == null) {
			printLineLog("No MLP specified");
			return;
		}
		
		JFileChooser fileChooser = new JFileChooser();
		fileChooser.setFileFilter(new FileNameExtensionFilter("Text Files", "txt"));
		fileChooser.setCurrentDirectory(new java.io.File("."));
		fileChooser.setDialogTitle("Specify a file to save");   
		 
		int userSelection = fileChooser.showSaveDialog(this);
		 
		if (userSelection == JFileChooser.APPROVE_OPTION) {
		    File fileToSave = fileChooser.getSelectedFile();
		    String path = fileToSave.getAbsolutePath() + ".txt";
			FileUtils.writeToFile(mlp, path);
			printStatus(mlp, "Saved network to " + path);
		}
	}

	private void newMLP() {
	    NewMLPPanel newMLPPanel = new NewMLPPanel();
		int result = JOptionPane.showConfirmDialog(null, newMLPPanel,
                "New MLP", JOptionPane.OK_CANCEL_OPTION,
                JOptionPane.PLAIN_MESSAGE);
          if (result == JOptionPane.OK_OPTION) {
        	MLP mlpNew = newMLPPanel.getMLP(); 
        	if (mlpNew != null) {
        		mlp = mlpNew;
	        	showInfo();
	  			printStatus(mlp, "Created network");
        	}
        	else
        		printLineLog("Failed to create MLP");
          }
	}

	private void inputDataSet() {
		if (mlp == null) {
			printLineLog("No MLP specified");
			return;
		}
		try {
			printStatus(mlp, "Started test on " + dataSet.getName());
			int nCorrect = mlp.test(dataSet);
			printStatus(mlp, "Finished test on " + dataSet.getName());
			printStatus(mlp, "Classified " + nCorrect + " correct of " + dataSet.getSize() + " (" + nCorrect * 100.0 / dataSet.getSize() + "%)");
		}
		catch (Exception e) {
			printStatus(mlp, e.getMessage());
		}
	}

	private void inputEntry() {
		if (mlp == null) {
			printLineLog("No MLP specified");
			return;
		}
		try {
			DataEntry entry = dataSet.getEntry(dataSetIndex);
			mlp.feedThrough(entry.getInput(), entry.getExpected());
			double cost = mlp.getCost();
			printStatus(mlp, "Best guess: " + mlp.mostLikely(mlp.getOutput()) + ", cost: " + Double.toString(cost));
		}
		catch (Exception e) {
			printStatus(mlp, e.getMessage());
		}
	}

	private void printStatus(MLP mlp, String s) {
		printLineLog(mlp.getName() + ": " + s);
	}
	
	private void printLineLog(String s) {
		SwingUtilities.invokeLater(new Runnable()
	    {
	        @Override
	        public void run()
	        {
	    		outputArea.append(s + "\n");
	    		outputArea.setCaretPosition(outputArea.getText().length());
	        }
	    });
	}
	
	private void updateIndexField() {
		dataIndexField.setValue(dataSetIndex + 1);
		sizeLabel.setText(" of " + dataSet.getSize());
	}
	
	private void nextData() {
		if (dataSetIndex < dataSet.getSize() - 1)
			dataSetIndex++;

		updateIndexField();
		
		if (dataSet.hasImage(dataSetIndex))
			imagePanel.setImage(dataSet.getImage(dataSetIndex));
	}

	private void indexEdit() {
		dataSetIndex = Utils.constrain((int) ((long) dataIndexField.getValue()), 1, dataSet.getSize()) - 1;
		updateIndexField();
		
		if (dataSet.hasImage(dataSetIndex))
			imagePanel.setImage(dataSet.getImage(dataSetIndex));
	}
	
	private void prevData() {
		if (dataSetIndex > 0)
			dataSetIndex--;
		
		updateIndexField();
		
		if (dataSet.hasImage(dataSetIndex))
			imagePanel.setImage(dataSet.getImage(dataSetIndex));
	}

	private void selectData(String data) {
		dataSet = dataSets.get(data);
		StringBuilder s = new StringBuilder();
		s.append(dataSet.getName() + "\n");
		s.append(dataSet.getSize() + "\n");
		
		dataArea.setText(s.toString());
		if (dataSet.hasImage(dataSetIndex))
			imagePanel.setImage(dataSet.getImage(dataSetIndex));
		
		dataSetIndex = 0;
		updateIndexField();
	}

	private void loadMLP() {
		JFileChooser fc = new JFileChooser();
		fc.setAcceptAllFileFilterUsed(false);
		fc.addChoosableFileFilter(new FileNameExtensionFilter("Text files", "txt"));
		fc.setCurrentDirectory(new File(System.getProperty("user.dir")));
		int returnVal = fc.showOpenDialog(this);

		if (returnVal == JFileChooser.APPROVE_OPTION) {
			mlp = FileUtils.readFromFile(fc.getSelectedFile().getPath());
			showInfo();
			printStatus(mlp, "Loaded network");
		}
	}
	
	private void showInfo() {
		StringBuilder s = new StringBuilder();
		s.append("Name: " + mlp.getName() + "\n");
		s.append("Learning rate: " + mlp.getLearningRate() + "\n");
		s.append("Input size: " + mlp.getInput().size() + "\n");
		for (int i = 0; i < mlp.getLayers().length; i++) {
			Layer l = mlp.getLayer(i);
			s.append("Layer " + (i + 1) + ":\n");
			s.append("w: ["+ l.getWeights().nRow() + "x" + l.getWeights().nCol() + "]\n");
			s.append("b: ["+ l.getBiases().size() + "x1]\n");
			s.append("Activation function: "+ l.getAFun().code() + "\n");
		}
		infoArea.setText(s.toString());
	}
}