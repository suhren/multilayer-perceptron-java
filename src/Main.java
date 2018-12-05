import java.io.File;
import java.io.FileNotFoundException;
import java.util.ArrayList;
import java.util.List;
import java.util.Scanner;

/**
 * The main entry point of the program.
 * @author Adam
 */
public class Main {

	public static void main(String[] args) {
		System.out.println("Hello World!");
		String filePath = "TestMLP.txt";
		MLP network = new MLP(readFromFile(filePath));
		System.out.print(network.toString() + "\n");
		network.feedThrough(new double[] {1.0, 1.0, 1.0, 1.0});
		System.out.print(network.toString());
	}
	
	private static Layer[] readFromFile(String filePath) {
		File file = new File(filePath);
		Scanner sc = null;
		try {
			sc = new Scanner(file);
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		}
		
		List<Integer> dim = new ArrayList<Integer>();
		Scanner scHeader = new Scanner(sc.nextLine());
		while (scHeader.hasNextInt())
			dim.add(scHeader.nextInt());

		Layer[] layers = new Layer[dim.size() - 1];
		
		for (int i = 0; i < dim.size()-1; i++) {
			int nRow = dim.get(i+1);
			int nCol = dim.get(i);
			layers[i] = new Layer(nRow, nCol);
			double[] data = new double[nCol];
			for (int row = 0; row < nRow; row++) {
				for (int col = 0; col < nCol; col++)
					data[col] = sc.nextDouble();
				layers[i].w[row] = data;
				layers[i].b[row] = sc.nextDouble();
			}
		}

		return layers;
	}
}