package com.mlp.program;

import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.PrintWriter;
import java.io.RandomAccessFile;
import java.io.UnsupportedEncodingException;
import java.nio.ByteBuffer;
import java.nio.channels.FileChannel;
import java.util.Scanner;

import com.mlp.data.MNIST;
import com.mlp.math.Utils;
import com.mlp.network.Layer;
import com.mlp.network.MLP;

/**
 * https://github.com/jeffgriffith/mnist-reader/blob/master/src/main/java/mnist/MnistReader.java
 * @author Adam
 *
 */
public class FileUtils {
	public static final int LABEL_FILE_MAGIC_NUMBER = 2049;
	public static final int IMAGE_FILE_MAGIC_NUMBER = 2051;
	
	public static void writeToFile(MLP network, String filePath) {
		PrintWriter writer = null;
		try {
			writer = new PrintWriter(filePath, "UTF-8");
		} catch (FileNotFoundException | UnsupportedEncodingException e) {
			e.printStackTrace();
		}
		
		writer.write(network.getName() + "\n");
		writer.write(network.getLearningRate() + "\n");
		writer.write(network.getInput().size() + "\n");
		
		for (Layer l : network.getLayers())
			writer.write(l.getWeights().nRow() + " ");
		writer.write("\n");
		for (Layer l : network.getLayers())
			writer.write(l.getAFun().code() + "\n");
		
		for (Layer l : network.getLayers())
			for (int row = 0; row < l.getWeights().nRow(); row++) {
				for (int col = 0; col < l.getWeights().nCol(); col++)
					writer.write(l.getWeights().get(row, col) + " ");
				writer.write(l.getBiases().get(row) + "\n");
			}
		writer.close();
	}
	
	public static MLP readFromFile(String filePath) {
		File file = new File(filePath);
		Scanner sc = null;
		try {
			sc = new Scanner(file);
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		} catch (Exception e) {
			e.printStackTrace();
		}
		
		double mlpRate = Double.parseDouble(sc.nextLine());
		int inputs = Integer.parseInt(sc.nextLine());
		
		String[] lStrings = sc.nextLine().split(" ");
		int[] dim = new int[lStrings.length];
		for (int i = 0; i < lStrings.length; i++)
			dim[i] = Integer.parseInt(lStrings[i]);
		
		for (int i = 0; i < dim.length; i++)
			lStrings[i] = sc.nextLine();

		Layer[] layers = new Layer[dim.length];
		
		for (int i = 0; i < dim.length; i++) {
			int nRow = dim[i];
			int nCol = inputs;
			layers[i] = new Layer(nRow, nCol, Utils.AFunLibrary.fromCode(lStrings[i]));
			double[] data = new double[nCol];
			for (int row = 0; row < nRow; row++) {
				for (int col = 0; col < nCol; col++)
					data[col] = sc.nextDouble();
				layers[i].getWeights().setRow(row, data);
				layers[i].getBiases().set(row, sc.nextDouble());
			}
			inputs = nRow;
		}

		String name = file.getName();
		name = name.substring(0, name.lastIndexOf('.'));
		return new MLP(name, mlpRate, layers);
	}
	
	public static MNIST readMNIST(String dataSetName, String labelPath, String imagePath) {
		return new MNIST(dataSetName, getMNISTLabels(labelPath), getMNISTImages(imagePath));
	}
	
	public static int[] getMNISTLabels(String file) {

		ByteBuffer bb = loadMNISTFileToByteBuffer(file);

		assertMagicNumber(LABEL_FILE_MAGIC_NUMBER, bb.getInt());

		int numLabels = bb.getInt();
		int[] labels = new int[numLabels];

		for (int i = 0; i < numLabels; ++i)
			labels[i] = bb.get() & 0xFF; // To unsigned

		return labels;
	}
	
	public static int[][][] getMNISTImages(String infile) {
		ByteBuffer bb = loadMNISTFileToByteBuffer(infile);

		assertMagicNumber(IMAGE_FILE_MAGIC_NUMBER, bb.getInt());

		int numImages = bb.getInt();
		int numRows = bb.getInt();
		int numColumns = bb.getInt();
		int[][][] images = new int[numImages][][];

		for (int i = 0; i < numImages; i++)
			images[i] = readImage(numRows, numColumns, bb);

		return images;
	}

	private static int[][] readImage(int numRows, int numCols, ByteBuffer bb) {
		int[][] image = new int[numRows][];
		for (int row = 0; row < numRows; row++)
			image[row] = readRow(numCols, bb);
		return image;
	}
	
	private static int[] readRow(int numCols, ByteBuffer bb) {
		int[] row = new int[numCols];
		for (int col = 0; col < numCols; ++col)
			row[col] = bb.get() & 0xFF; // To unsigned
		return row;
	}
	
	public static void assertMagicNumber(int expectedMagicNumber, int magicNumber) {
		if (expectedMagicNumber != magicNumber) {
			switch (expectedMagicNumber) {
				case LABEL_FILE_MAGIC_NUMBER:
					throw new RuntimeException("This is not a label file.");
				case IMAGE_FILE_MAGIC_NUMBER:
					throw new RuntimeException("This is not an image file.");
				default:
					throw new RuntimeException(String.format("Expected magic number %d, found %d", expectedMagicNumber, magicNumber));
			}
		}
	}
	
	private static ByteBuffer loadMNISTFileToByteBuffer(String file) {
		return ByteBuffer.wrap(loadMNISTFile(file));
	}
	
	private static byte[] loadMNISTFile(String file) {
		try {
			RandomAccessFile f = new RandomAccessFile(file, "r");
			FileChannel chan = f.getChannel();
			long fileSize = chan.size();
			ByteBuffer bb = ByteBuffer.allocate((int) fileSize);
			chan.read(bb);
			bb.flip();
			ByteArrayOutputStream out = new ByteArrayOutputStream();
			for (int i = 0; i < fileSize; i++)
				out.write(bb.get());
			chan.close();
			f.close();
			return out.toByteArray();
		}
		catch (Exception e) {
			throw new RuntimeException(e);
		}
	}
}
